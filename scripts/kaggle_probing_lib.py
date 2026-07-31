# -*- coding: utf-8 -*-
"""M4 probing library: file-level-resumable hidden-state extraction + the
generalized A/M/CB probe battery. A plain importable module (not notebook
cells) so the logic is unit-testable without a GPU or a real model
download (tests/test_kaggle_probing_lib.py uses a tiny mock "model"). The
Kaggle notebook (notebooks/m4_probing.ipynb) clones this repo and imports
this module directly.

Two independent pieces:

1. Extraction (`extract_g2p`). Each (tokenizer, category) combo is a single
   forward pass per word over at most ~3,000 words — minutes, not hours,
   even for an 8B model on a T4 — so restart risk isn't "a run dies
   mid-file," it's "you come back in a later session (a new week's GPU
   quota, a disconnect between combos) and don't want to redo already-
   finished combos." That only needs FILE-level resumability: skip a combo
   entirely if its output already exists and looks complete. (An earlier
   draft of this module added row-level mid-file checkpointing; dropped as
   unneeded complexity for this scale — see docs/DEVELOPMENT_LOG.md.)

2. The probe battery (`run_probe_battery`). Upstream's train_probe_*.py
   scripts (liaodisen/Tokenization-Phonology) are hardcoded to exactly two
   conditions (their good/bad Hamming-distance split) and specific
   English-dataset filenames — not reusable as-is for our three-way A/M/CB
   split (Spec C.4). This is a from-scratch reimplementation of the SAME
   protocol (RidgeCV for regression targets / LogisticRegression for
   classification, 10 seeds, 80/20 split, optional label/embedding control
   tasks per Spec A.6) but generalized to however many categories are
   present, with pairwise one-sided t-tests computed automatically
   (Spec C.6.2's A > M > CB comparison) instead of just the one comparison
   upstream supports.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Sequence

__all__ = ["extraction_done", "extract_g2p", "run_probe_battery"]


# ----------------------------------------------------------------------------
# 1. Extraction, file-level resumable
# ----------------------------------------------------------------------------

def _embeddings_finite(embeddings) -> bool:
    import numpy as np
    arr = np.asarray(embeddings, dtype="float64")
    return bool(np.isfinite(arr).all())


def extraction_done(out_dir: str, n_layers: int) -> bool:
    """True if out_dir already has all n_layers layer_{i}.pkl files with a
    nonzero word count AND finite embedding values — good enough to trust
    and skip re-extracting. A checkpoint containing inf/nan (e.g. from a
    T5-family model run in plain fp16, a known instability — T5 was
    trained in bfloat16/fp32 and plain fp16 inference can overflow
    internal activations) is treated as NOT done, same as a corrupted
    pickle: the combo gets automatically redone rather than silently
    trusted or requiring the user to manually delete the bad checkpoint."""
    if not os.path.isdir(out_dir):
        return False
    for li in range(n_layers):
        path = os.path.join(out_dir, f"layer_{li}.pkl")
        if not os.path.exists(path):
            return False
    try:
        for li in range(n_layers):
            with open(os.path.join(out_dir, f"layer_{li}.pkl"), "rb") as f:
                data = pickle.load(f)
            if len(data["word"]) == 0:
                return False
            if not _embeddings_finite(data["embeddings"]):
                return False
        return True
    except Exception:
        return False           # corrupted/partial write from a killed session: redo it


def extract_g2p(extract_fn, rows: Sequence[dict], out_dir: str, n_layers: int,
                storage_dtype: str = "float32") -> str:
    """Extract last-token hidden states for every layer, for every row's
    `word`. Skips entirely (returns immediately) if out_dir already looks
    complete per extraction_done() — that's the whole resumability story
    for this scale (see module docstring): re-running the notebook after
    any kind of restart just skips finished combos instead of redoing them.

    extract_fn(word: str) -> List[np.ndarray], one array per layer (layer 0
        = embedding layer). Caller owns model/tokenizer lifetime — load
        once outside the loop over combos, this function only drives one
        combo's extraction loop.
    rows: dicts with 'word' (str), 'phon_vec' (list[int]), 'syllables'
        (list[list[str]], only its len() is used).
    storage_dtype: defaults to float32, NOT float16 — an earlier version
        defaulted to float16 to save disk space and it turned out unsafe:
        even with correctly-finite fp32 model output (see extraction_done's
        docstring re: T5's fp16-compute overflow, a separate issue), some
        BanglaT5 hidden states exceed float16's ~65504 max on real data
        (word বীর্য, layer 4, hit exactly this on the first real run).
        float32 costs 2x the disk for a problem this dataset's scale
        (<=3,000 words/combo) doesn't make worth optimizing around.
    Writes layer_{i}.pkl in upstream's own format (word/embeddings/
    phon_vecs/syllable_count keys) so downstream code doesn't care whether
    they came from this or the original script.
    """
    import numpy as np

    if extraction_done(out_dir, n_layers):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)

    words, phon_vecs, syllable_counts = [], [], []
    embeddings_by_layer: Dict[int, list] = {}
    for row in rows:
        layer_embeds = extract_fn(row["word"])
        if not embeddings_by_layer:
            embeddings_by_layer = {li: [] for li in range(len(layer_embeds))}
        for li, emb in enumerate(layer_embeds):
            if not np.isfinite(emb).all():
                raise RuntimeError(
                    f"non-finite values in extract_fn output for word={row['word']!r}, "
                    f"layer={li} — before any storage casting, so this is the model's own "
                    f"output, not a downcast artifact. If this is a T5-family model, it's "
                    f"almost certainly running in plain float16 (T5 was trained in "
                    f"bfloat16/fp32 and plain fp16 inference commonly overflows internal "
                    f"activations) — load it in float32 instead.")
            cast = np.asarray(emb, dtype=storage_dtype)
            if not np.isfinite(cast).all():
                raise RuntimeError(
                    f"word={row['word']!r}, layer={li}: finite in the model's native dtype "
                    f"but overflowed casting to {storage_dtype} for storage. Use a wider "
                    f"storage_dtype (float32 is already the default; if you're already on "
                    f"float32 and still hitting this, the model's native values are "
                    f"exceeding ~3.4e38, which would be its own separate bug).")
            embeddings_by_layer[li].append(cast)
        words.append(row["word"])
        phon_vecs.append(row["phon_vec"])
        syllable_counts.append(len(row["syllables"]))

    for li, embs in embeddings_by_layer.items():
        with open(os.path.join(out_dir, f"layer_{li}.pkl"), "wb") as f:
            pickle.dump({"word": words, "embeddings": embs,
                        "phon_vecs": phon_vecs, "syllable_count": syllable_counts}, f)
    return out_dir


# ----------------------------------------------------------------------------
# 2. Probe battery: generalized A / M / CB comparison
# ----------------------------------------------------------------------------

def run_probe_battery(X_by_cat: Dict[str, Sequence], y_by_cat: Dict[str, Sequence],
                      probe: str = "ridge", n_seeds: int = 10, seed_start: int = 10,
                      control_label: bool = False, control_embed: bool = False) -> dict:
    """Same protocol as upstream's train_probe_{g2p,syl,rhyme}.py (RidgeCV
    for regression targets, LogisticRegression for classification, 10
    seeds, 80/20 split) but for however many categories are in X_by_cat/
    y_by_cat (2 for rhyme-style A-vs-CB, 3 for the full A/M/CB comparison),
    with every pairwise one-sided t-test computed automatically instead of
    upstream's fixed single comparison.

    probe: 'ridge' (multi-output or scalar regression target, e.g. phon_vec
        or syllable_count) or 'logistic' (binary classification, e.g. rhyme
        label).
    control_label / control_embed: Spec A.6's random-label / random-
        embedding controls. Mutually exclusive; control_label wins if both
        are set (matches upstream's elif chain).
    """
    import numpy as np
    from scipy.stats import ttest_rel
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split

    if set(X_by_cat) != set(y_by_cat):
        raise ValueError("X_by_cat and y_by_cat must have the same category keys")

    scores_by_cat: Dict[str, List[float]] = {cat: [] for cat in X_by_cat}
    for seed in range(seed_start, seed_start + n_seeds):
        rng = np.random.RandomState(seed)
        for cat in X_by_cat:
            X = np.asarray(X_by_cat[cat], dtype=float)
            y = np.asarray(y_by_cat[cat], dtype=float)
            if control_label:
                y = rng.randn(*y.shape) if probe == "ridge" else rng.randint(
                    0, max(len(set(y.tolist())), 2), size=len(y))
            if control_embed:
                X = rng.randn(*X.shape)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed)

            if probe == "ridge":
                std = y.std(axis=0) if y.ndim > 1 else y.std()
                std = np.where(std == 0, 1, std) if y.ndim > 1 else (std or 1)
                mean = y.mean(axis=0) if y.ndim > 1 else y.mean()
                y_train_n = (y_train - mean) / std
                y_test_n = (y_test - mean) / std
                model = RidgeCV(alphas=[10, 100, 500, 1000, 2000])
                model.fit(X_train, y_train_n)
                score = r2_score(y_test_n, model.predict(X_test))
            elif probe == "logistic":
                model = LogisticRegression(max_iter=1000, C=10)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
            else:
                raise ValueError(f"unknown probe type: {probe!r}")
            scores_by_cat[cat].append(score)

    result = {
        "scores": scores_by_cat,
        "mean": {c: float(np.mean(s)) for c, s in scores_by_cat.items()},
        "std": {c: float(np.std(s)) for c, s in scores_by_cat.items()},
    }
    pairwise = {}
    cats = list(scores_by_cat)
    for a in cats:
        for b in cats:
            if a == b:
                continue
            if len(scores_by_cat[a]) > 1 and len(scores_by_cat[a]) == len(scores_by_cat[b]):
                t_stat, p = ttest_rel(scores_by_cat[a], scores_by_cat[b])
                pairwise[f"{a}>{b}"] = float(p / 2 if t_stat > 0 else 1 - p / 2)
    result["pairwise_one_sided_p"] = pairwise
    return result

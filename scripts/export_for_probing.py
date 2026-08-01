# -*- coding: utf-8 -*-
"""Export our frozen task datasets into the CSV shapes consumed by the forked
probing harness (external/Tokenization-Phonology, Liao & Shi 2026), and split
by the A / M / CB word categories (Spec C.4) in place of their good/bad split.

Mapping to the upstream repo (see docs/probing_integration.md for the full
walkthrough — the fork itself is NOT committed here, it's a plain
`git clone`, reproducible from that doc):
  data/tasks/g2p.jsonl          -> data/probing_export/g2p_{tokenizer}_{A,M,CB}.csv
                                    columns: word, phon_vec, syllables
                                    (phon_vec = phoneme indices, padded to the
                                    dataset max length with 0 — their
                                    train_probe_g2p.py consumes this shape
                                    directly; our max is reported below vs.
                                    their English max of 8)
  data/tasks/syllable_count_word.jsonl -> reuses the same CSV (syllables col)
  data/task3a_rhyme_pairs.jsonl -> data/probing_export/rhyme_pairs.csv
                                    (tokenizer-agnostic flat file, columns:
                                    word1, word2, label)
                                 -> data/probing_export/rhyme_{tokenizer}_{A,M,CB}.csv
                                    (per-tokenizer A/M/CB split, same columns
                                    — see categorize_pair()'s docstring for how
                                    two per-word categories become one
                                    per-pair category)

At M4 time, copy (or symlink) data/probing_export/*.csv into the forked
repo's probing/data/ directory before running generate_embedding.py.

This script does NOT call any GPU code — no model weights are loaded, only
forward passes (probing/generate_embedding.py) are milestone M4 and run on
Kaggle T4s. It DOES load tokenizers (CPU-only, lightweight, same pattern as
scripts/compute_metrics.py) to compute the per-tokenizer rhyme-pair GTAD/
STAD categories, since data/task3a_rhyme_pairs.jsonl (unlike g2p.jsonl)
doesn't carry precomputed per-tokenizer tokenization fields.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/probing_export")
TASKS = Path("data/tasks")
TOKENIZERS = ["llama3", "gpt2", "byt5", "tigerllm", "banglat5"]
M_THRESHOLD = 0.25   # Spec C.4: Syllable-misaligned category threshold


def categorize(gtad, stad):
    if gtad is None or stad is None:
        return None
    if gtad > 0:
        return "CB"
    if stad > M_THRESHOLD:
        return "M"
    return "A"


_CATEGORY_RANK = {"A": 0, "M": 1, "CB": 2}


def categorize_pair(cat1, cat2):
    """Combine two per-word A/M/CB categories into one category for a rhyme
    PAIR (Task 3a's probing prompt is f"{word1} {word2}" -- a single joint
    embedding, so there's one category per pair, not per word). Takes the
    worse (more misaligned) of the two, on the reasoning that if either
    word's tokenization is broken, the joint hidden state is compromised
    regardless of the other word's alignment. None if either word's
    category is undefined (quarantined tokenization or single-akshara
    STAD=None)."""
    if cat1 is None or cat2 is None:
        return None
    return cat1 if _CATEGORY_RANK[cat1] >= _CATEGORY_RANK[cat2] else cat2


def build_phoneme_vocab(items):
    vocab = Counter()
    for it in items:
        vocab.update(it["phonemes"])
    # deterministic index assignment: most frequent first (ties broken alphabetically)
    ordered = sorted(vocab, key=lambda p: (-vocab[p], p))
    return {p: i + 1 for i, p in enumerate(ordered)}   # 0 reserved for padding


def main():
    items = [json.loads(l) for l in open(TASKS / "g2p.jsonl", encoding="utf-8")]
    pad_len = max(len(it["phonemes"]) for it in items)
    vocab = build_phoneme_vocab(items)
    print(f"[export] {len(items)} g2p items, phoneme vocab size={len(vocab)}, "
          f"pad_len={pad_len} (English arpabet reference: 8)")

    OUT.mkdir(parents=True, exist_ok=True)
    for tk in TOKENIZERS:
        rows = {"A": [], "M": [], "CB": []}
        n_missing = 0
        for it in items:
            block = it["tokenization"].get(tk)
            if not block or block.get("quarantined"):
                n_missing += 1
                continue
            cat = categorize(block["gtad"], block["stad"])
            if cat is None:
                n_missing += 1
                continue
            idx = [vocab[p] for p in it["phonemes"]] + [0] * (pad_len - len(it["phonemes"]))
            rows[cat].append({
                "word": it["orth"],
                "phon_vec": idx,
                "syllables": it["syllables_phonemic"],
                "ipa": it["ipa"],
            })
        for cat, rs in rows.items():
            path = OUT / f"g2p_{tk}_{cat}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["word", "phon_vec", "syllables", "ipa"])
                w.writeheader()
                for r in rs:
                    w.writerow(r)
            print(f"  {path.name}: {len(rs)} words")
        if n_missing:
            print(f"  ({tk}: {n_missing} items skipped — quarantined/missing tokenization)")

    with open(TASKS / "phoneme_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "pad_len": pad_len}, f, ensure_ascii=False, indent=2)
    print(f"[export] wrote phoneme vocab -> data/tasks/phoneme_vocab.json")

    rhyme = [json.loads(l) for l in open("data/task3a_rhyme_pairs.jsonl", encoding="utf-8")]
    rp = OUT / "rhyme_pairs.csv"
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2", "label"])
        w.writeheader()
        for it in rhyme:
            w.writerow({"word1": it["orth1"], "word2": it["orth2"], "label": it["label"]})
    print(f"[export] wrote {rp.name}: {len(rhyme)} pairs")

    build_rhyme_category_export(rhyme, TOKENIZERS)


def build_rhyme_category_export(rhyme_items, tokenizer_names):
    """Per-tokenizer A/M/CB split for Task 3a rhyme pairs (the join flagged
    as not-yet-built in notebooks/m4_probing.ipynb's final cell). Unlike
    g2p.jsonl, task3a_rhyme_pairs.jsonl carries no precomputed per-tokenizer
    tokenization field, so GTAD/STAD are computed fresh here per word per
    tokenizer (same primitives as scripts/compute_metrics.py), then combined
    per pair via categorize_pair()."""
    from bangla_phonology import gtad, stad_bn
    from src.tokenizer_adapter import (MisalignedTokenizationError,
                                       load_tokenizers, real_token_byte_spans)

    tokenizers = load_tokenizers(tokenizer_names)
    if not tokenizers:
        print("[export] no tokenizer could be loaded -- skipping rhyme A/M/CB export")
        return

    def word_category(orth, phonemes, tk):
        try:
            spans = real_token_byte_spans(orth, tk)
        except MisalignedTokenizationError:
            return None
        g = gtad(orth, spans)
        s = stad_bn(orth, phonemes, spans)
        return categorize(g.gtad, s.stad)

    for tk_name, tk in tokenizers.items():
        rows = {"A": [], "M": [], "CB": []}
        n_excluded = 0
        for it in rhyme_items:
            cat1 = word_category(it["orth1"], it["phonemes1"], tk)
            cat2 = word_category(it["orth2"], it["phonemes2"], tk)
            cat = categorize_pair(cat1, cat2)
            if cat is None:
                n_excluded += 1
                continue
            rows[cat].append({"word1": it["orth1"], "word2": it["orth2"], "label": it["label"]})
        for cat, rs in rows.items():
            path = OUT / f"rhyme_{tk_name}_{cat}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["word1", "word2", "label"])
                w.writeheader()
                for r in rs:
                    w.writerow(r)
            print(f"  rhyme_{tk_name}_{cat}.csv: {len(rs)} pairs")
        if n_excluded:
            print(f"  ({tk_name}: {n_excluded} pairs excluded — quarantined/undefined "
                  f"category for at least one word)")


if __name__ == "__main__":
    main()

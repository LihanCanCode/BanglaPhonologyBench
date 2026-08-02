# -*- coding: utf-8 -*-
"""M5 regression + A/M/CB breakdown (Spec C.6 steps 2 and 5): does a word's
tokenization misalignment (GTAD, STAD_bn, rho) predict whether a zero-shot
model gets it right, beyond what frequency/etymology alone would predict?

g2p and syllable_count reuse precomputed per-word `tokenization.{tokenizer}.
{gtad,stad}` and `rho` fields already stored in `data/tasks/*.jsonl` -- no
new metric computation needed. schwa_deletion.jsonl and
task3a_rhyme_pairs.jsonl do NOT carry those precomputed fields (see
scripts/export_for_probing.py's docstring for why -- task3a needs a joint
two-word prompt's tokenization, not a single word's), so those two compute
GTAD/STAD/rho fresh via the same primitives scripts/compute_metrics.py and
export_for_probing.py use (`bangla_phonology.gtad`/`stad_bn` +
`src.tokenizer_adapter.real_token_byte_spans`), which means loading the
actual tokenizer (network access, lightweight -- tokenizer files only, no
model weights) and computing `wordfreq.zipf_frequency` for log_freq since
neither file carries a precomputed zipf either.

Two things this script produces per task, both from the SAME per-word
correct/incorrect array:
  1. A/M/CB accuracy breakdown (categorize() from export_for_probing.py,
     Spec C.4's fixed A/M/CB definition -- not re-derived here).
  2. A logistic regression of `correct` ~ gtad + stad + rho + log_freq, via
     a small hand-rolled IRLS (Newton's method) implementation using only
     the stdlib `math` module -- deliberately not adding statsmodels/
     sklearn as a new dependency for one script. This gets standardized
     coefficients + McFadden's pseudo-R^2, but NOT p-values/standard
     errors/a real inference table -- flagged as a real limitation, not
     silently upgraded to more than what's actually computed here. If the
     thesis needs proper significance testing on this regression, redo it
     with statsmodels.Logit before publication; treat these coefficients
     as directional/exploratory only.

Run: `python -X utf8 scripts/analyze_regression.py syllable_count <path-to-raw-completions.jsonl>`
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from export_for_probing import categorize, categorize_pair  # noqa: E402
from zeroshot_lib import parse_schwa_answer, parse_yes_no  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO_ROOT, "data", "tasks")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

TASK_GOLD_PATHS = {
    "g2p": os.path.join(TASKS_DIR, "g2p.jsonl"),
    "syllable_count": os.path.join(TASKS_DIR, "syllable_count_word.jsonl"),
    "schwa_deletion": os.path.join(TASKS_DIR, "schwa_deletion.jsonl"),
    "rhyme_awareness": os.path.join(REPO_ROOT, "data", "task3a_rhyme_pairs.jsonl"),
}

_INT_RE = re.compile(r"-?\d+")


def parse_syllable_count(text: str):
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def parse_g2p(text: str) -> str:
    return text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(ast.literal_eval(line))
    return rows


def _build_precomputed(task, gold_by_id, raw_by_id, tokenizer):
    """g2p / syllable_count: everything needed is already stored per word."""
    records = []
    for i, raw in raw_by_id.items():
        g = gold_by_id.get(i)
        if g is None:
            continue
        block = g["tokenization"].get(tokenizer)
        if not block or block.get("quarantined"):
            continue
        cat = categorize(block["gtad"], block["stad"])
        if cat is None:
            continue

        if task == "syllable_count":
            pred = parse_syllable_count(raw)
            correct = int(pred == g["syllable_count"])
        elif task == "g2p":
            pred = parse_g2p(raw)
            correct = int(pred == g["ipa"])
        else:
            raise ValueError(f"unexpected task={task!r} in precomputed path")

        records.append({
            "id": i, "correct": correct, "gtad": block["gtad"], "stad": block["stad"],
            "rho": g["rho"], "log_freq": g.get("zipf", 0.0), "category": cat,
        })
    return records


def _build_schwa(gold_by_id, raw_by_id, tokenizer):
    """schwa_deletion.jsonl has no precomputed tokenization/rho/zipf -- compute
    fresh, same primitives as scripts/compute_metrics.py."""
    from bangla_phonology import gtad, stad_bn
    from src.tokenizer_adapter import (MisalignedTokenizationError,
                                       load_tokenizers, real_token_byte_spans)
    from wordfreq import zipf_frequency

    tokenizers = load_tokenizers([tokenizer])
    tk = tokenizers[tokenizer]

    records = []
    for i, raw in raw_by_id.items():
        g = gold_by_id.get(i)
        if g is None:
            continue
        try:
            spans = real_token_byte_spans(g["orth"], tk)
        except MisalignedTokenizationError:
            continue
        gt = gtad(g["orth"], spans)
        st = stad_bn(g["orth"], g["phonemes"], spans, syllables=g["syllables_phonemic"])
        cat = categorize(gt.gtad, st.stad)
        if cat is None:
            continue

        gold_vec = g["schwa_vector"]
        pred_vec = parse_schwa_answer(raw, len(gold_vec))
        correct = int(None not in pred_vec and pred_vec == gold_vec)

        records.append({
            "id": i, "correct": correct, "gtad": gt.gtad, "stad": (st.stad or 0.0),
            "rho": st.rho, "log_freq": zipf_frequency(g["orth"], "bn"), "category": cat,
            "etym": g.get("etym"),
        })
    return records


def _build_rhyme_awareness(gold_by_id, raw_by_id, tokenizer):
    """task3a_rhyme_pairs.jsonl has no precomputed per-word tokenization
    either -- compute GTAD/STAD/rho fresh per word, combine into one
    per-PAIR feature set via the worse-of-two-words reasoning
    (export_for_probing.categorize_pair uses the same reasoning for the
    category label; here gtad/stad/rho are numeric, so "worse" = max)."""
    from bangla_phonology import gtad, stad_bn
    from src.tokenizer_adapter import (MisalignedTokenizationError,
                                       load_tokenizers, real_token_byte_spans)
    from wordfreq import zipf_frequency

    tokenizers = load_tokenizers([tokenizer])
    tk = tokenizers[tokenizer]

    def word_metrics(orth, phonemes):
        try:
            spans = real_token_byte_spans(orth, tk)
        except MisalignedTokenizationError:
            return None
        gt = gtad(orth, spans)
        st = stad_bn(orth, phonemes, spans)
        cat = categorize(gt.gtad, st.stad)
        if cat is None:
            return None
        return {"gtad": gt.gtad, "stad": st.stad or 0.0, "rho": st.rho, "category": cat}

    records = []
    for i, raw in raw_by_id.items():
        g = gold_by_id.get(i)
        if g is None:
            continue
        m1 = word_metrics(g["orth1"], g["phonemes1"])
        m2 = word_metrics(g["orth2"], g["phonemes2"])
        if m1 is None or m2 is None:
            continue
        cat = categorize_pair(m1["category"], m2["category"])
        if cat is None:
            continue

        pred = parse_yes_no(raw)
        correct = int(pred is not None and pred == g["label"])

        records.append({
            "id": i, "correct": correct,
            "gtad": max(m1["gtad"], m2["gtad"]), "stad": max(m1["stad"], m2["stad"]),
            "rho": (m1["rho"] + m2["rho"]) / 2,
            "log_freq": (zipf_frequency(g["orth1"], "bn") + zipf_frequency(g["orth2"], "bn")) / 2,
            "category": cat,
        })
    return records


def build_dataset(task, completions_path, tokenizer="tigerllm"):
    gold_rows = load_jsonl(TASK_GOLD_PATHS[task])
    gold_by_id = {r["id"]: r for r in gold_rows}
    raw_by_id = {r["id"]: r["raw_output"] for r in load_jsonl(completions_path)}

    if task in ("g2p", "syllable_count"):
        return _build_precomputed(task, gold_by_id, raw_by_id, tokenizer)
    if task == "schwa_deletion":
        return _build_schwa(gold_by_id, raw_by_id, tokenizer)
    if task == "rhyme_awareness":
        return _build_rhyme_awareness(gold_by_id, raw_by_id, tokenizer)
    raise ValueError(f"scoring not implemented for task={task!r} yet")


def amcb_breakdown(records):
    from collections import defaultdict
    buckets = defaultdict(lambda: [0, 0])
    for r in records:
        buckets[r["category"]][0] += r["correct"]
        buckets[r["category"]][1] += 1
    return {cat: (correct / n, n) for cat, (correct, n) in buckets.items()}


def logistic_regression(records, feature_names):
    """Minimal IRLS logistic regression, no new dependency. Standardizes
    features first (mean 0, sd 1) so coefficients are comparable in
    magnitude -- same reasoning as kaggle_probing_lib's ridge-probe
    standardization fix (see docs/DEVELOPMENT_LOG.md, "Standardize features
    before the logistic (rhyme) probe fit")."""
    n = len(records)
    y = [r["correct"] for r in records]
    raw_X = [[r[f] for f in feature_names] for r in records]

    means = [sum(col) / n for col in zip(*raw_X)]
    sds = [max(1e-9, math.sqrt(sum((v - m) ** 2 for v in col) / n))
           for col, m in zip(zip(*raw_X), means)]
    X = [[1.0] + [(v - m) / s for v, m, s in zip(row, means, sds)] for row in raw_X]

    k = len(feature_names) + 1
    beta = [0.0] * k

    def sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    for _ in range(50):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for row, yi in zip(X, y):
            z = sum(b * x for b, x in zip(beta, row))
            p = sigmoid(z)
            w = max(p * (1 - p), 1e-6)
            err = yi - p
            for a in range(k):
                grad[a] += row[a] * err
                for b in range(k):
                    hess[a][b] -= w * row[a] * row[b]
        # Newton step: beta -= hess^-1 @ grad, via Gaussian elimination on -hess
        A = [[-hess[a][b] for b in range(k)] + [grad[a]] for a in range(k)]
        for col in range(k):
            piv = max(range(col, k), key=lambda r: abs(A[r][col]))
            A[col], A[piv] = A[piv], A[col]
            if abs(A[col][col]) < 1e-12:
                continue
            for r in range(k):
                if r == col:
                    continue
                factor = A[r][col] / A[col][col]
                for c in range(col, k + 1):
                    A[r][c] -= factor * A[col][c]
        delta = [A[i][k] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(k)]
        beta = [b + d for b, d in zip(beta, delta)]
        if max(abs(d) for d in delta) < 1e-8:
            break

    ll = 0.0
    for row, yi in zip(X, y):
        z = sum(b * x for b, x in zip(beta, row))
        p = min(max(sigmoid(z), 1e-9), 1 - 1e-9)
        ll += yi * math.log(p) + (1 - yi) * math.log(1 - p)
    p_null = sum(y) / n
    p_null = min(max(p_null, 1e-9), 1 - 1e-9)
    ll_null = n * (p_null * math.log(p_null) + (1 - p_null) * math.log(1 - p_null))
    mcfadden_r2 = 1 - ll / ll_null if ll_null != 0 else float("nan")

    return {"intercept": beta[0], "coefs": dict(zip(feature_names, beta[1:])),
            "mcfadden_r2": mcfadden_r2, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASK_GOLD_PATHS))
    ap.add_argument("completions_path")
    ap.add_argument("--tokenizer", default="tigerllm")
    args = ap.parse_args()

    records = build_dataset(args.task, args.completions_path, args.tokenizer)
    print(f"{len(records)} records usable (had a defined A/M/CB category)\n")

    print("A/M/CB accuracy breakdown:")
    for cat, (acc, n) in sorted(amcb_breakdown(records).items()):
        print(f"  {cat}: acc={acc:.3f}  n={n}")

    print("\nLogistic regression: correct ~ gtad + stad + rho + log_freq (standardized coefs)")
    reg = logistic_regression(records, ["gtad", "stad", "rho", "log_freq"])
    print(f"  n={reg['n']}  McFadden pseudo-R^2={reg['mcfadden_r2']:.4f}")
    print(f"  intercept={reg['intercept']:.3f}")
    for name, coef in reg["coefs"].items():
        direction = "lower accuracy" if coef < 0 else "higher accuracy"
        print(f"  {name:10s} coef={coef:+.3f}  (higher {name} -> {direction})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"regression_{args.task}_{args.tokenizer}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "tokenizer": args.tokenizer,
                    "amcb": {c: {"acc": a, "n": n} for c, (a, n) in amcb_breakdown(records).items()},
                    "regression": reg}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

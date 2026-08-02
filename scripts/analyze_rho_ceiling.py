# -*- coding: utf-8 -*-
"""M5 rho-ceiling analysis (Spec C.6.5's regression, narrow single-variable
slice): does a word's rho (fraction of its phonemic syllable boundaries
that fall INSIDE an akshara, per bangla_phonology.py -- a property of the
script itself, not any tokenizer) predict how often a zero-shot model gets
its syllable count right?

Hypothesis (script-topology ceiling, not just "tokenizers are bad"): higher
rho -> the syllable structure is less recoverable from the orthographic
akshara boundaries alone -> lower accuracy, for BOTH a naive orthography-only
method (scripts/build_baselines.py's segment_aksharas-count baseline) and a
zero-shot LLM working from spelling. Reuses `rho` already precomputed and
stored per word in data/tasks/syllable_count_word.jsonl -- no new metric
computation needed.

Run: `python -X utf8 scripts/analyze_rho_ceiling.py <path-to-raw-completions-jsonl>`
where the completions file is the {id, raw_output} JSONL a zero-shot run
wrote to checkpoints/zeroshot/syllable_count_bn_<model>.jsonl.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bangla_phonology import normalize_bn, segment_aksharas  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(REPO_ROOT, "data", "tasks", "syllable_count_word.jsonl")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

_INT_RE = re.compile(r"-?\d+")


def parse_syllable_count(text: str):
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def rho_bin(rho: float) -> str:
    if rho == 0:
        return "0.0 (no ceiling)"
    if rho <= 0.34:
        return "0.01-0.34"
    if rho < 1.0:
        return "0.35-0.99"
    return "1.0 (fully misaligned)"


BIN_ORDER = ["0.0 (no ceiling)", "0.01-0.34", "0.35-0.99", "1.0 (fully misaligned)"]


def load_jsonl(path):
    """Tolerates both real JSONL and Python-dict-literal lines (single
    quotes) -- some completions files got hand-downloaded/renamed with a
    stray print() repr instead of json.dumps output."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("completions_path", help="checkpoints/zeroshot/syllable_count_bn_<model>.jsonl")
    ap.add_argument("--model-name", default="model")
    args = ap.parse_args()

    gold_rows = load_jsonl(GOLD_PATH)
    gold_by_id = {r["id"]: r for r in gold_rows}

    raw_by_id = {r["id"]: r["raw_output"] for r in load_jsonl(args.completions_path)}
    matched = [i for i in raw_by_id if i in gold_by_id]
    print(f"matched {len(matched)}/{len(gold_by_id)} gold rows against completions")

    bins = {b: {"n": 0, "model_correct": 0, "baseline_correct": 0} for b in BIN_ORDER}
    for i in matched:
        g = gold_by_id[i]
        rho = g["rho"]
        gold_count = g["syllable_count"]

        pred = parse_syllable_count(raw_by_id[i])
        baseline_pred = len(segment_aksharas(normalize_bn(g["orth"])))

        b = rho_bin(rho)
        bins[b]["n"] += 1
        if pred == gold_count:
            bins[b]["model_correct"] += 1
        if baseline_pred == gold_count:
            bins[b]["baseline_correct"] += 1

    print(f"\nSyllable-count accuracy by rho bin ({args.model_name} vs. naive akshara-count baseline)\n")
    print(f"{'rho bin':24s} {'n':>6s} {args.model_name+' acc':>14s} {'baseline acc':>14s}")
    rows_out = []
    for b in BIN_ORDER:
        d = bins[b]
        if d["n"] == 0:
            continue
        model_acc = d["model_correct"] / d["n"]
        baseline_acc = d["baseline_correct"] / d["n"]
        print(f"{b:24s} {d['n']:6d} {model_acc:14.3f} {baseline_acc:14.3f}")
        rows_out.append({"rho_bin": b, "n": d["n"], "model_acc": model_acc,
                          "baseline_acc": baseline_acc, "model_name": args.model_name})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"rho_ceiling_syllable_count_{args.model_name}.csv")
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rho_bin", "n", "model_acc", "baseline_acc", "model_name"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Score a full set of downloaded zero-shot raw-completion files locally
(no GPU/model needed) via scripts/zeroshot_lib.py's scorers, producing one
authoritative zeroshot_summary_{model}.csv -- the same shape
notebooks/m5_zeroshot.ipynb's section 6 scoring cell writes on Kaggle, for
when you'd rather score locally against files already downloaded than
re-run that cell.

Run: `python -X utf8 scripts/score_zeroshot_results.py --model tigerllm
      --g2p results/g2p_results.csv
      --syllable_count results/syllable_count_results.csv
      --rhyme_awareness results/rhyme_awareness_results.csv
      --rhyme_generation results/rhyme_generation_results.csv
      --schwa_deletion results/schwa_deletion_results.csv`
Any task flag can be omitted if you don't have that file yet.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import zeroshot_lib as zs  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO_ROOT, "data", "tasks")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

TASK_GOLD_PATHS = {
    "g2p": os.path.join(TASKS_DIR, "g2p.jsonl"),
    "syllable_count": os.path.join(TASKS_DIR, "syllable_count_word.jsonl"),
    "rhyme_awareness": os.path.join(REPO_ROOT, "data", "task3a_rhyme_pairs.jsonl"),
    "rhyme_generation": os.path.join(REPO_ROOT, "data", "task3b_rhyme_generation.jsonl"),
    "schwa_deletion": os.path.join(TASKS_DIR, "schwa_deletion.jsonl"),
}

TASK_RUNNERS = {
    "g2p": zs.run_g2p,
    "syllable_count": zs.run_syllable_count,
    "rhyme_awareness": zs.run_rhyme_awareness,
    "rhyme_generation": zs.run_rhyme_generation,
    "schwa_deletion": zs.run_schwa,
}


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


def score_task(task, completions_path, lang="bn"):
    gold_rows = load_jsonl(TASK_GOLD_PATHS[task])
    raw_by_id = {r["id"]: r["raw_output"] for r in load_jsonl(completions_path)}
    scored_rows = [r for r in gold_rows if r["id"] in raw_by_id]

    def canned_generate_fn(_prompt, _row_iter=iter(scored_rows)):
        return raw_by_id[next(_row_iter)["id"]]

    return TASK_RUNNERS[task](scored_rows, canned_generate_fn, lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tigerllm")
    ap.add_argument("--lang", default="bn")
    for task in TASK_GOLD_PATHS:
        ap.add_argument(f"--{task}", help=f"path to {task} raw completions file")
    args = ap.parse_args()

    results = []
    for task in TASK_GOLD_PATHS:
        path = getattr(args, task)
        if not path:
            print(f"{task}: skipped (no --{task} path given)")
            continue
        s = score_task(task, path, args.lang)
        results.append(s)
        print(f"{task:18s} {args.lang}  n={s.n:5d} parsed={s.n_parsed:5d}  {s.metrics}")

    out_path = os.path.join(RESULTS_DIR, f"zeroshot_summary_{args.model}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "lang", "n", "n_parsed", "metric", "value"])
        for s in results:
            for k, v in s.metrics.items():
                w.writerow([s.task, s.lang, s.n, s.n_parsed, k, v])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

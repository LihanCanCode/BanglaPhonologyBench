# -*- coding: utf-8 -*-
"""Export stratified CSV review sheets for the M3 human annotation pass.

Sized for a SOLO annotator in one project (not a funded multi-annotator
lab): full coverage where the pool is already small or precious (all rhyme
pairs, all heuristic-foreign words), stratified sampling elsewhere. Total
review load is ~700 rows across three sheets — a few focused sittings, not
weeks of work.

Every sheet uses the same two-column convention, kept consistent on purpose
so there's no ambiguity between "I looked at this" and "I agree with it":
  - `reviewed`: set to TRUE once you've looked at the row. Blank = not
    reviewed yet — apply_annotations.py only touches rows with reviewed=TRUE.
  - a `*_corrected` column: leave BLANK if you agree with the predicted
    value: fill it in ONLY to override.

Writes (data/annotation/):
  etym_review.csv   ~350 words: all 53 heuristic="foreign" + a random
                     stratified sample of tatsama/tadbhava (balanced across
                     freq_bucket and, where available, the TigerLLM A/M/CB
                     category — the one tokenizer with a real 3-way split,
                     Spec C.4). Columns: id, orth, ipa, freq_bucket,
                     tigerllm_category, etym_heuristic, etym_corrected,
                     reviewed, notes.
  task3a_rhyme_review.csv  all 400 pairs from data/task3a_rhyme_pairs.jsonl
                     (src/rhyme.py — proper open/closed-syllable rime, NOT
                     the retired data/tasks/rhyme_pairs.jsonl). Columns: id,
                     orth1, orth2, ipa1, ipa2, rime1, rime2, assonance_key1,
                     assonance_key2, neg_type, predicted_label, freq_bucket1,
                     freq_bucket2, label_corrected, reviewed, notes.
  schwa_review.csv  ~160 words stratified across the four schwa
                     environments (final/post_conjunct/conjunct/medial).
                     Columns: id, orth, aksharas, schwa_vector,
                     schwa_environments, vector_corrected, reviewed, notes.

You fill these in by hand (Excel, LibreOffice, Google Sheets — all handle
UTF-8 CSV fine) and then run scripts/apply_annotations.py to merge your
edits back into data/tasks/*.jsonl.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SEED = 13
OUT = Path("data/annotation")
TASKS = Path("data/tasks")


def load(name):
    return [json.loads(l) for l in open(TASKS / name, encoding="utf-8")]


def tigerllm_category(it, threshold=0.25):
    block = it["tokenization"].get("tigerllm")
    if not block or block.get("quarantined") or block["gtad"] is None:
        return ""
    if block["gtad"] > 0:
        return "CB"
    if block["stad"] is not None and block["stad"] > threshold:
        return "M"
    return "A"


def build_etym_sheet(rng, target_tatsama=150, target_tadbhava=150):
    items = load("g2p.jsonl")
    foreign = [it for it in items if it["etym"] == "foreign"]
    tatsama = [it for it in items if it["etym"] == "tatsama"]
    tadbhava = [it for it in items if it["etym"] == "tadbhava"]
    rng.shuffle(tatsama)
    rng.shuffle(tadbhava)
    sample = foreign + tatsama[:target_tatsama] + tadbhava[:target_tadbhava]
    rng.shuffle(sample)

    rows = []
    for it in sample:
        rows.append({
            "id": it["id"], "orth": it["orth"], "ipa": it["ipa"],
            "freq_bucket": it["freq_bucket"],
            "tigerllm_category": tigerllm_category(it),
            "etym_heuristic": it["etym"],
            # fill: tatsama / tadbhava / foreign / deshi — ONLY if you disagree.
            # deshi = indigenous substrate vocabulary, NOT descended from
            # Sanskrit at all (unlike tadbhava); the heuristic never guesses
            # it since it has no orthographic signature — see Spec A.3.5.
            "etym_corrected": "",
            "reviewed": "",           # set TRUE once you've looked at this row
            "notes": "",
        })
    path = OUT / "etym_review.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[etym_review] {len(rows)} rows "
          f"(foreign={len(foreign)}, tatsama={min(target_tatsama,len(tatsama))}, "
          f"tadbhava={min(target_tadbhava,len(tadbhava))}) -> {path}")


def build_rhyme_sheet():
    """Task 3a rhyme pairs (src/rhyme.py + scripts/build_rhyme_dataset.py),
    NOT the retired data/tasks/rhyme_pairs.jsonl / rhyme_review.csv pair —
    those were superseded (see docs/annotation_guide.md)."""
    items = [json.loads(l) for l in open("data/task3a_rhyme_pairs.jsonl", encoding="utf-8")]
    rows = []
    for it in items:
        rows.append({
            "id": it["id"], "orth1": it["orth1"], "orth2": it["orth2"],
            "ipa1": "".join(it["phonemes1"]), "ipa2": "".join(it["phonemes2"]),
            "rime1": "".join(it["rime1"]) if it["rime1"] else "",
            "rime2": "".join(it["rime2"]) if it["rime2"] else "",
            "assonance_key1": it["assonance_key1"] or "",
            "assonance_key2": it["assonance_key2"] or "",
            "neg_type": it["neg_type"] or "",
            "predicted_label": it["label"],
            "freq_bucket1": it["freq_bucket1"], "freq_bucket2": it["freq_bucket2"],
            "label_corrected": "",   # fill: 1 (rhymes) / 0 (doesn't) — ONLY if you disagree
            "reviewed": "",           # set TRUE once you've looked at this row
            "notes": "",
        })
    path = OUT / "task3a_rhyme_review.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[task3a_rhyme_review] {len(rows)} rows -> {path}")


def build_schwa_sheet(rng, per_env=40):
    items = load("schwa_deletion.jsonl")
    by_env = {}
    for it in items:
        key = tuple(sorted(set(it["schwa_environments"])))
        by_env.setdefault(key, []).append(it)

    buckets = {"final": [], "post_conjunct": [], "conjunct": [], "medial": []}
    for it in items:
        for env in set(it["schwa_environments"]):
            if env in buckets:
                buckets[env].append(it)

    picked, seen_ids = [], set()
    for env, pool in buckets.items():
        rng.shuffle(pool)
        n = 0
        for it in pool:
            if it["id"] in seen_ids:
                continue
            picked.append(it)
            seen_ids.add(it["id"])
            n += 1
            if n >= per_env:
                break

    rows = []
    for it in picked:
        rows.append({
            "id": it["id"], "orth": it["orth"],
            "aksharas": " ".join(it["grapheme_clusters"]),
            "schwa_vector": " ".join(str(b) for b in it["schwa_vector"]),
            "schwa_environments": " ".join(it["schwa_environments"]),
            "vector_corrected": "",  # fill: space-separated 0/1 — ONLY if you disagree
            "reviewed": "",           # set TRUE once you've looked at this row
            "notes": "",
        })
    path = OUT / "schwa_review.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[schwa_review] {len(rows)} rows (~{per_env}/environment) -> {path}")


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    build_etym_sheet(rng)
    build_rhyme_sheet()
    build_schwa_sheet(rng)


if __name__ == "__main__":
    main()

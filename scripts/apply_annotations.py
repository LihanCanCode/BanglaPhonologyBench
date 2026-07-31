# -*- coding: utf-8 -*-
"""Merge your edits in data/annotation/*.csv back into data/tasks/*.jsonl.

Only rows with reviewed=TRUE are applied — everything else is left exactly
as the heuristic/computed pipeline produced it, still annotation.verified=
false. Safe to re-run as you review more rows over multiple sittings; it
always re-derives from the CSVs, so partial progress is never lost.

Solo-annotator note (Spec A.4 Task 1 calls for 10% dual-annotation +
phoneme-level kappa): this project has one annotator, so no inter-annotator
agreement is computed. `annotation.annotators` is set to 1 throughout —
report this as a limitation, not silently upgrade it.

Usage: python scripts/apply_annotations.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ANN = Path("data/annotation")
TASKS = Path("data/tasks")


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_true(v):
    return str(v).strip().upper() in ("TRUE", "1", "YES", "Y")


def apply_etym():
    csv_path = ANN / "etym_review.csv"
    if not csv_path.exists():
        print("[etym] no etym_review.csv, skipping")
        return
    rows = read_csv(csv_path)
    reviewed = {r["id"]: r for r in rows if is_true(r["reviewed"])}
    if not reviewed:
        print(f"[etym] 0/{len(rows)} rows marked reviewed=TRUE, nothing to apply")
        return

    # id -> k index (g2p_00042 -> 42), to also patch syllable_count_word.jsonl
    k_of = {rid: int(rid.rsplit("_", 1)[1]) for rid in reviewed}
    orth_of_k = {}

    g2p = load_jsonl(TASKS / "g2p.jsonl")
    agree = correct = 0
    for it in g2p:
        r = reviewed.get(it["id"])
        if not r:
            continue
        final = r["etym_corrected"].strip() or r["etym_heuristic"]
        if r["etym_corrected"].strip():
            correct += 1
        else:
            agree += 1
        it["etym"] = final
        it["annotation"]["verified"] = True
        it["annotation"]["annotators"] = 1
        orth_of_k[k_of[it["id"]]] = (final, it["orth"])
    write_jsonl(TASKS / "g2p.jsonl", g2p)

    syl_path = TASKS / "syllable_count_word.jsonl"
    if syl_path.exists():
        syl = load_jsonl(syl_path)
        n_syl = 0
        for it in syl:
            k = int(it["id"].rsplit("_", 1)[1])
            if k in orth_of_k:
                final, orth = orth_of_k[k]
                if it["orth"] == orth:      # sanity: same word at same index
                    it["etym"] = final
                    it["annotation"]["verified"] = True
                    it["annotation"]["annotators"] = 1
                    n_syl += 1
        write_jsonl(syl_path, syl)
        print(f"[etym] propagated to {n_syl} matching syllable_count_word.jsonl items")

    # propagate to schwa_deletion.jsonl by orthography match (separate sample)
    schwa_path = TASKS / "schwa_deletion.jsonl"
    if schwa_path.exists():
        schwa = load_jsonl(schwa_path)
        by_orth = {orth: final for final, orth in orth_of_k.values()}
        n_schwa = 0
        for it in schwa:
            if it["orth"] in by_orth:
                it["etym"] = by_orth[it["orth"]]
                n_schwa += 1
        write_jsonl(schwa_path, schwa)
        print(f"[etym] propagated to {n_schwa} matching schwa_deletion.jsonl items")

    print(f"[etym] applied {len(reviewed)}/{len(rows)} reviewed rows "
          f"(agreed with heuristic: {agree}, corrected: {correct}, "
          f"heuristic accuracy on reviewed sample: {agree/len(reviewed):.1%})")


def apply_rhyme():
    """Task 3a rhyme pairs (data/task3a_rhyme_pairs.jsonl, src/rhyme.py) —
    NOT the retired data/tasks/rhyme_pairs.jsonl."""
    csv_path = ANN / "task3a_rhyme_review.csv"
    if not csv_path.exists():
        print("[rhyme] no task3a_rhyme_review.csv, skipping")
        return
    rows = read_csv(csv_path)
    reviewed = {r["id"]: r for r in rows if is_true(r["reviewed"])}
    if not reviewed:
        print(f"[rhyme] 0/{len(rows)} rows marked reviewed=TRUE, nothing to apply")
        return

    rhyme_path = Path("data/task3a_rhyme_pairs.jsonl")
    items = load_jsonl(rhyme_path)
    agree = correct = 0
    for it in items:
        r = reviewed.get(it["id"])
        if not r:
            continue
        override = r["label_corrected"].strip()
        final = int(override) if override else it["label"]
        if override:
            correct += 1
        else:
            agree += 1
        it["label"] = final
        it["annotation"]["verified"] = True
        it["annotation"]["annotators"] = 1
    write_jsonl(rhyme_path, items)
    print(f"[rhyme] applied {len(reviewed)}/{len(rows)} reviewed rows "
          f"(agreed: {agree}, corrected: {correct}, "
          f"predicted-label accuracy on reviewed sample: {agree/len(reviewed):.1%})")


def apply_schwa():
    csv_path = ANN / "schwa_review.csv"
    if not csv_path.exists():
        print("[schwa] no schwa_review.csv, skipping")
        return
    rows = read_csv(csv_path)
    reviewed = {r["id"]: r for r in rows if is_true(r["reviewed"])}
    if not reviewed:
        print(f"[schwa] 0/{len(rows)} rows marked reviewed=TRUE, nothing to apply")
        return

    items = load_jsonl(TASKS / "schwa_deletion.jsonl")
    agree = correct = 0
    for it in items:
        r = reviewed.get(it["id"])
        if not r:
            continue
        override = r["vector_corrected"].strip()
        if override:
            it["schwa_vector"] = [int(x) for x in override.split()]
            correct += 1
        else:
            agree += 1
        it["annotation"]["verified"] = True
        it["annotation"]["annotators"] = 1
    write_jsonl(TASKS / "schwa_deletion.jsonl", items)
    print(f"[schwa] applied {len(reviewed)}/{len(rows)} reviewed rows "
          f"(agreed: {agree}, corrected: {correct}, "
          f"predicted-vector accuracy on reviewed sample: {agree/len(reviewed):.1%})")


def main():
    apply_etym()
    apply_rhyme()
    apply_schwa()
    print("\nNote: single annotator (this project) — annotation.annotators=1, "
          "no inter-annotator agreement / kappa computed. Document as a "
          "limitation (Spec A.4 Task 1 assumes 2 annotators for kappa).")


if __name__ == "__main__":
    main()

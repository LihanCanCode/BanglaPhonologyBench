# -*- coding: utf-8 -*-
"""Heuristic etymology candidate-tagger (Spec A.3.5): tatsama / tadbhava / foreign.

This is CANDIDATE GENERATION ONLY (Spec A.1: "rule-based ... outputs always
human-verified before entering gold"). It writes a first-pass guess into the
`etym` field of the frozen task JSONL files; `annotation.verified` stays
False until a human confirms it via scripts/apply_annotations.py.

Heuristic (orthographic cues only — no etymological dictionary available):
  FOREIGN candidate  — contains the অ্যা/এ্যা digraph (the only way to write
                       /æ/, which has no native Bangla vowel letter and is
                       almost always an English loan signal: ব্যাংক, ক্যামেরা),
                       OR an initial স্ট/স্ক/স্প cluster (Greco-Latin/English
                       loan clusters absent from tadbhava: স্টেশন, স্কুল).
  TATSAMA candidate  — contains a genuine consonant conjunct (hasanta-joined
                       C-H-C chain; native tadbhava vocabulary is
                       overwhelmingly CV(C) with no conjuncts, Spec B.2),
                       OR visarga ঃ, OR ঋ/ৃ (ri-kar) — both are
                       Sanskrit-orthography-only devices.
  TADBHAVA (default)  — no conjunct, no tatsama-only diacritics, no foreign
                       markers: simple CV(C) native-looking structure.

This will mis-tag plenty of words (e.g. Perso-Arabic loans look tadbhava-
simple; some tatsama compounds have no conjunct). That's expected — it only
needs to save you typing on the ~70% of cases the heuristic gets right, per
Spec A.1's "candidate generation ONLY" contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import HASANTA, RI_KAR, VISARGA, segment_aksharas

FOREIGN_CLUSTERS = ("স্ট", "স্ক", "স্প")


def has_conjunct(orth: str) -> bool:
    return any(HASANTA in cl and len(cl) > 1 and cl[-1] != HASANTA
               for cl in segment_aksharas(orth))


def guess_etym(orth: str) -> str:
    if "অ্যা" in orth or "এ্যা" in orth or any(c in orth for c in FOREIGN_CLUSTERS):
        return "foreign"
    if has_conjunct(orth) or VISARGA in orth or RI_KAR in orth:
        return "tatsama"
    return "tadbhava"


def tag_file(path: Path) -> int:
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = 0
    for it in items:
        word = it.get("orth") or it.get("word1")   # rhyme_pairs has no single orth
        if word is None:
            continue
        guess = guess_etym(word)
        if it.get("etym") != guess:
            it["etym"] = guess
            n += 1
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return n


def main():
    for name in ("g2p.jsonl", "syllable_count_word.jsonl", "schwa_deletion.jsonl"):
        path = Path("data/tasks") / name
        n = tag_file(path)
        print(f"[tag_etymology_heuristic] {name}: tagged {n} items")


if __name__ == "__main__":
    main()

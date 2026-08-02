# -*- coding: utf-8 -*-
"""M5 baselines (Spec A.6 "Baselines" row) — non-LLM reference points to compare
zero-shot model performance against. Every baseline here is deliberately naive
(no lexicon lookup for G2P/syllables, no learned parameters) so that beating it
is a low bar; they exist to calibrate "is the LLM doing something, or nothing."

Four baselines, one per task:
  - G2P              : `naive_g2p` — direct grapheme->phoneme table walk over
                        `segment_aksharas` clusters. No silent-letter deletion,
                        no gemination/assimilation, no conjunct-specific rules
                        (contrast `bangla_phonology.align`, which has all of
                        that and is what produced the gold labels — never
                        reused here, that would make the "baseline" the gold
                        labeler). Only schwa handling: word-final inherent
                        vowel dropped, medial inherent vowel kept as /o/ —
                        the same simplification as the schwa baseline below,
                        applied uniformly rather than positionally-informed.
  - Syllable count   : `len(segment_aksharas(word))` — "one syllable per
                        orthographic akshara." Already named in CLAUDE.md as
                        the segmenter's fallback syllable proxy. Undercounts
                        whenever a coda consonant cluster spans two aksharas
                        (geminate splits, Cluster codas) since those still
                        syllabify as 2 syllables phonemically.
  - Rhyme (3a)       : "dictionary rime lookup" — uses the pair's PRECOMPUTED
                        `rime1`/`rime2` fields (derived from lexicon
                        pronunciation via `src.rhyme.rime`), predicts
                        rhyme iff rime1 == rime2. This is a ceiling-style
                        baseline: it assumes perfect pronunciation knowledge
                        (a dictionary lookup) and asks only "given that, is
                        rhyme detection easy?" (it is, by construction —
                        that's expected, not a bug; the interesting question
                        for M5 is whether an LLM can recover it from spelling
                        alone, without the dictionary).
  - Schwa deletion   : majority-rule heuristic, literally the one named in
                        Spec A.6: delete word-final schwa, keep all others
                        (medial/conjunct/post_conjunct majority-keep on this
                        dataset too — see docs/DEVELOPMENT_LOG.md M5 section
                        for the empirical per-environment majority check that
                        confirmed this before hardcoding it).
  - Rhyme (3b)       : same "dictionary lookup" ceiling idea as 3a's rhyme
                        baseline, for the generation task — pick any 5 words
                        from the prompt's precomputed `gold_rhymes` (already
                        the full anti-leakage-filtered rime group, see
                        scripts/build_rhyme_generation_dataset.py). This is
                        trivially 100% BY CONSTRUCTION (the "prediction" is
                        drawn from the gold set itself) — not a meaningful
                        floor, just the same "if you have a dictionary, this
                        task is easy" ceiling reference 3a's baseline gives,
                        so an LLM's zero-shot success@5 has something to be
                        compared against besides nothing.

Run: `python -X utf8 scripts/build_baselines.py`
Writes `results/baselines_summary.csv` (one row per task, headline metrics)
and prints a human-readable report.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bangla_phonology import (ANUSVARA, CONSONANTS, INDEP_VOWELS, KHANDA_TA,
                               MATRAS, normalize_bn, segment_aksharas)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO_ROOT, "data", "tasks")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ----------------------------------------------------------------------------
# Naive grapheme-to-phoneme table (isolated-letter pronunciation; no context
# rules). Matches project IPA conventions (dental t̪/d̪, retroflex ʈ/ɖ,
# affricates tʃ/dʒ, ফ -> f) so PER is measured on a like-for-like symbol set.
# ----------------------------------------------------------------------------
CONSONANT_PHON = {
    "ক": "k", "খ": "kʰ", "গ": "g", "ঘ": "gʱ", "ঙ": "ŋ",
    "চ": "tʃ", "ছ": "tʃʰ", "জ": "dʒ", "ঝ": "dʒʱ", "ঞ": "n",
    "ট": "ʈ", "ঠ": "ʈʰ", "ড": "ɖ", "ঢ": "ɖʱ", "ণ": "n",
    "ত": "t̪", "থ": "t̪ʰ", "দ": "d̪", "ধ": "d̪ʱ", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bʱ", "ম": "m",
    "য": "dʒ", "র": "r", "ল": "l", "শ": "ʃ", "ষ": "ʃ", "স": "s", "হ": "h",
    "ড়": "ɽ", "ঢ়": "ɽʱ", "য়": "j",
}
INDEP_VOWEL_PHON = {
    "অ": "ɔ", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
    "ঋ": "ri", "এ": "e", "ঐ": "oi̯", "ও": "o", "ঔ": "ou̯",
}
MATRA_PHON = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u",
    "ৃ": "ri", "ে": "e", "ৈ": "oi̯", "ো": "o", "ৌ": "ou̯",
}
INHERENT_VOWEL = "o"


def naive_g2p(word: str) -> List[str]:
    """Cluster-by-cluster direct table lookup. See module docstring for the
    deliberate omissions (no assimilation, no gemination, no silent letters)."""
    w = normalize_bn(word)
    clusters = segment_aksharas(w)
    out: List[str] = []
    for idx, cl in enumerate(clusters):
        is_last = idx == len(clusters) - 1
        if cl[0] in INDEP_VOWELS:
            out.append(INDEP_VOWEL_PHON.get(cl[0], cl[0]))
        else:
            for ch in cl:
                if ch in CONSONANTS:
                    out.append(CONSONANT_PHON.get(ch, ch))
            matra = next((ch for ch in cl if ch in MATRAS), None)
            if matra is not None:
                out.append(MATRA_PHON[matra])
            elif not is_last:
                out.append(INHERENT_VOWEL)
            # is_last and no matra: word-final inherent vowel deleted (naive
            # schwa rule, matches the schwa baseline's "delete word-final")
        if ANUSVARA in cl:
            out.append("ŋ")
        if KHANDA_TA in cl:
            out.append("t̪")
    return out


def syllable_baseline(word: str) -> int:
    return len(segment_aksharas(normalize_bn(word)))


def schwa_baseline(environments: Sequence[str]) -> List[int]:
    return [0 if env == "final" else 1 for env in environments]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_g2p_baseline():
    rows = load_jsonl(os.path.join(TASKS_DIR, "g2p.jsonl"))
    total_per, exact, n = 0.0, 0, 0
    for r in rows:
        gold = r["phonemes"]
        pred = naive_g2p(r["orth"])
        if not gold:
            continue
        per = levenshtein(pred, gold) / len(gold)
        total_per += per
        exact += int(pred == gold)
        n += 1
    return {"task": "g2p", "n": n, "metric": "mean_PER", "value": total_per / n,
            "secondary_metric": "exact_match", "secondary_value": exact / n}


def run_syllable_baseline():
    rows = load_jsonl(os.path.join(TASKS_DIR, "syllable_count_word.jsonl"))
    exact, abs_err, n = 0, 0, 0
    for r in rows:
        gold = r["syllable_count"]
        pred = syllable_baseline(r["orth"])
        exact += int(pred == gold)
        abs_err += abs(pred - gold)
        n += 1
    return {"task": "syllable_count", "n": n, "metric": "exact_match_acc", "value": exact / n,
            "secondary_metric": "MAE", "secondary_value": abs_err / n}


def run_schwa_baseline():
    rows = load_jsonl(os.path.join(TASKS_DIR, "schwa_deletion.jsonl"))
    exact, pos_correct, pos_total, n = 0, 0, 0, 0
    for r in rows:
        gold = r["schwa_vector"]
        pred = schwa_baseline(r["schwa_environments"])
        if len(pred) != len(gold):
            continue
        exact += int(pred == gold)
        pos_correct += sum(1 for p, g in zip(pred, gold) if p == g)
        pos_total += len(gold)
        n += 1
    return {"task": "schwa_deletion", "n": n, "metric": "per_position_acc", "value": pos_correct / pos_total,
            "secondary_metric": "exact_vector_match", "secondary_value": exact / n}


def run_rhyme_baseline():
    rows = load_jsonl(os.path.join(REPO_ROOT, "data", "task3a_rhyme_pairs.jsonl"))
    tp = fp = tn = fn = 0
    for r in rows:
        pred = int(r["rime1"] is not None and r["rime1"] == r["rime2"])
        gold = r["label"]
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else float("nan")
    return {"task": "rhyme_awareness", "n": n, "metric": "accuracy", "value": acc,
            "secondary_metric": "F1", "secondary_value": f1}


def run_rhyme_generation_baseline(k: int = 5):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from rhyme import mean_success_at_k  # noqa: E402

    rows = load_jsonl(os.path.join(REPO_ROOT, "data", "task3b_rhyme_generation.jsonl"))
    all_candidates = [r["gold_rhymes"][:k] for r in rows]
    all_gold = [set(r["gold_rhymes"]) for r in rows]
    score = mean_success_at_k(all_candidates, all_gold, k=k)
    return {"task": "rhyme_generation", "n": len(rows), "metric": f"success_at_{k}", "value": score,
            "secondary_metric": "n/a", "secondary_value": float("nan")}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = [run_g2p_baseline(), run_syllable_baseline(),
               run_schwa_baseline(), run_rhyme_baseline(),
               run_rhyme_generation_baseline()]

    out_path = os.path.join(RESULTS_DIR, "baselines_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "n", "metric", "value",
                                          "secondary_metric", "secondary_value"])
        w.writeheader()
        w.writerows(results)

    print("Naive baselines (Spec A.6) — see script docstring for what each one does/skips\n")
    for r in results:
        print(f"  {r['task']:16s} n={r['n']:5d}  {r['metric']}={r['value']:.4f}"
              f"   {r['secondary_metric']}={r['secondary_value']:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Ingest the Google language-resources bn-BD pronunciation lexicon.

Steps (Spec A.3 / C.4):
  1. NFC-normalize orthography; drop entries with Bangla-character ratio < 0.9.
  2. Map lexicon phoneme notation -> our IPA set (data/phoneme_map.tsv);
     merge glide symbols (i̯ u̯ e̯ o̯) with the adjacent vowel into diphthongs.
  3. Keep the lexicon's own syllabification (the ' . ' marks) as gold.
  4. Run the akshara segmenter + orthography->phoneme aligner on every entry.
  5. Report: totals, aligner success rate, failure categories (+30-sample file),
     syllable-count agreement between our rule-based syllabifier and the lexicon.

Output: data/lexicon_clean.tsv
  columns: orth, ipa_phonemes (space-sep, ' . ' syllable marks kept),
           aksharas (space-sep), syllables (from lexicon), syllable_count,
           aligner_ok

Usage: python scripts/ingest_lexicon.py [--lexicon data/google_bn_lexicon.tsv]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import align, normalize_bn, segment_aksharas, syllabify

BENGALI_RE = re.compile(r"[ঀ-৿‌‍]")
GLIDES = {"i̯", "u̯", "e̯", "o̯"}
VOWSET = set("aeiouæɔ")


def load_phoneme_map(path: str) -> dict:
    mp = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            mp[cols[0]] = cols[1]
    return mp


def is_vowel(p: str) -> bool:
    b = unicodedata.normalize("NFD", p).replace("̯", "")
    return bool(b) and b[0] in VOWSET


def map_syllables(trans: str, mp: dict):
    """Lexicon transcription -> list of syllables (lists of our IPA phonemes).
    Glide symbols merge with the adjacent vowel (offglide preferred, onglide
    otherwise). Returns (syllables, error_or_None)."""
    syllables = []
    for syl_str in trans.split("."):
        syms = syl_str.split()
        if not syms:
            return None, "empty syllable"
        out = []
        for s in syms:
            if s not in mp:
                return None, f"unknown symbol {s!r}"
            p = mp[s]
            if p in GLIDES and out and is_vowel(out[-1]):
                out[-1] = out[-1] + p              # offglide: e + i̯ -> ei̯
            else:
                out.append(p)
        # onglide: i̯ u  ->  i̯u  (glide left at position k, vowel right after)
        merged = []
        for p in out:
            if merged and merged[-1] in GLIDES and is_vowel(p):
                merged[-1] = merged[-1] + p
            else:
                merged.append(p)
        if any(p in GLIDES for p in merged):
            return None, f"unattached glide in {trans!r}"
        syllables.append(merged)
    return syllables, None


def bangla_ratio(word: str) -> float:
    if not word:
        return 0.0
    return sum(1 for ch in word if BENGALI_RE.match(ch)) / len(word)


def failure_category(note: str) -> str:
    if "nucleus expected" in note:
        return "nucleus_expected_for_indep_vowel"
    if "vowel where consonant expected" in note:
        return "vowel_where_consonant_expected"
    if "unconsumed" in note:
        return "unconsumed_phonemes"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/google_bn_lexicon.tsv")
    ap.add_argument("--map", default="data/phoneme_map.tsv")
    ap.add_argument("--out", default="data/lexicon_clean.tsv")
    ap.add_argument("--failures-out", default="data/aligner_failures_sample.tsv")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    mp = load_phoneme_map(args.map)
    stats = Counter()
    rows, failures = [], []
    syl_agree = syl_total = 0
    seen = set()

    with open(args.lexicon, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                stats["malformed_line"] += 1
                continue
            orth = normalize_bn(parts[0].strip())
            stats["total"] += 1
            if bangla_ratio(orth) < 0.9:
                stats["dropped_nonbangla"] += 1
                continue
            syllables, err = map_syllables(parts[1].strip(), mp)
            if err:
                stats["dropped_unmappable"] += 1
                continue
            phonemes = [p for s in syllables for p in s]
            key = (orth, " ".join(phonemes))
            if key in seen:                       # exact duplicates only;
                stats["dropped_duplicate"] += 1   # homographs w/ diff pron kept
                continue
            seen.add(key)

            aksharas = segment_aksharas(orth)
            a = align(orth, phonemes)
            stats["kept"] += 1
            stats["aligner_ok" if a.ok else "aligner_fail"] += 1
            if not a.ok:
                failures.append((orth, parts[1].strip(), " ".join(phonemes),
                                 failure_category(a.note), a.note))
                stats[f"fail::{failure_category(a.note)}"] += 1

            # syllable-count agreement: our rule-based syllabifier vs lexicon
            ours = syllabify(phonemes)
            syl_total += 1
            if len(ours) == len(syllables):
                syl_agree += 1
                if ours == syllables:
                    stats["syllabifier_exact_match"] += 1

            ipa_marked = " . ".join(" ".join(s) for s in syllables)
            rows.append("\t".join([
                orth, ipa_marked, " ".join(aksharas),
                ipa_marked, str(len(syllables)), str(a.ok),
            ]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("orth\tipa_phonemes\taksharas\tsyllables\tsyllable_count\taligner_ok\n")
        f.write("\n".join(rows) + "\n")

    rng = random.Random(args.seed)
    sample = rng.sample(failures, min(30, len(failures)))
    with open(args.failures_out, "w", encoding="utf-8", newline="\n") as f:
        f.write("orth\tlexicon_transcription\tmapped_ipa\tcategory\tnote\n")
        for row in sample:
            f.write("\t".join(row) + "\n")

    kept = stats["kept"] or 1
    print("=" * 70)
    print("LEXICON INGESTION REPORT — Google language-resources bn-BD (CC BY 4.0)")
    print("=" * 70)
    print(f"total entries read           : {stats['total']}")
    print(f"dropped: bangla ratio < 0.9  : {stats['dropped_nonbangla']}")
    print(f"dropped: unmappable phonemes : {stats['dropped_unmappable']}")
    print(f"dropped: exact duplicates    : {stats['dropped_duplicate']}")
    print(f"kept                         : {stats['kept']}")
    print(f"aligner ok                   : {stats['aligner_ok']}"
          f"  ({stats['aligner_ok'] / kept:.2%})")
    print(f"aligner failures             : {stats['aligner_fail']}")
    for k in sorted(stats):
        if k.startswith("fail::"):
            print(f"    {k[6:]:<34}: {stats[k]}")
    print(f"syllable-COUNT agreement     : {syl_agree}/{syl_total}"
          f"  ({syl_agree / max(syl_total, 1):.2%})   [our syllabifier vs lexicon]")
    print(f"syllable exact-match         : {stats['syllabifier_exact_match']}"
          f"  ({stats['syllabifier_exact_match'] / max(syl_total, 1):.2%})")
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"wrote {len(sample)} failure samples -> {args.failures_out}")


if __name__ == "__main__":
    main()

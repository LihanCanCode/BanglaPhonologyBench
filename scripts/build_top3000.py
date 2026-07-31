# -*- coding: utf-8 -*-
"""Build the top-N frequency wordlist for the descriptive analysis (Spec C.6.1).

Frequency source: `wordfreq` (rspeer), Bengali model — a blend that includes
Bangla Wikipedia among its corpora. (The paper's final frequency stratification
will use IndicCorp-bn per Spec A.3; wordfreq is the reproducible interim proxy.)

Keeps the most frequent words that (a) appear in data/lexicon_clean.tsv and
(b) passed the orthography->phoneme aligner. Homographs: first lexicon entry
wins (frequency lists are orthographic).

Output: data/wordlist_top3000.tsv  (orth <TAB> dotted phonemes <TAB> zipf)
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon_clean.tsv")
    ap.add_argument("--out", default="data/wordlist_top3000.tsv")
    ap.add_argument("-n", type=int, default=3000)
    ap.add_argument("--pool", type=int, default=200_000,
                    help="how many top-frequency candidates to scan")
    args = ap.parse_args()

    import wordfreq

    lex = {}
    with open(args.lexicon, encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            orth, ipa, aligner_ok = cols[0], cols[1], cols[5]
            if aligner_ok == "True" and orth not in lex:
                lex[orth] = ipa

    out_rows, seen = [], set()
    for w in wordfreq.top_n_list("bn", args.pool):
        w = unicodedata.normalize("NFC", w)
        if w in lex and w not in seen:
            seen.add(w)
            z = wordfreq.zipf_frequency(w, "bn")
            out_rows.append(f"{w}\t{lex[w]}\t{z:.2f}")
            if len(out_rows) >= args.n:
                break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_rows) + "\n")
    print(f"[build_top3000] wrote {len(out_rows)} words -> {args.out}"
          f"  (pool scanned: {args.pool}, lexicon coverage bottleneck if < n)")


if __name__ == "__main__":
    main()

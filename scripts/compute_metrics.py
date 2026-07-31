# -*- coding: utf-8 -*-
"""Compute GTAD (+ decomposition), STAD_bn, rho per word per tokenizer.

Input wordlist: TSV, one entry per line:  orth<TAB>phonemes
  - phonemes are space-separated IPA symbols (one phoneme per token);
  - an optional " . " between phonemes marks gold syllable boundaries
    (lexicon-style: "ʃ a n . t̪ o"); when present the gold syllabification is
    used for STAD instead of our rule-based syllabifier.

Output: tidy CSV, one row per (word, tokenizer).

Usage:
  python scripts/compute_metrics.py data/wordlist.tsv -o results/metrics.csv \
         --tokenizers llama3 gpt2 byt5 tigerllm banglat5
"""
from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import align, gtad, segment_aksharas, stad_bn
from src.tokenizer_adapter import (MisalignedTokenizationError,
                                   load_tokenizers, real_token_byte_spans)

FIELDS = ["word", "tokenizer", "tokens", "n_tokens", "n_boundaries",
          "gtad", "gtad_byte", "gtad_matra", "gtad_conjunct",
          "stad", "rho", "m_aksharas", "n_phonemes", "aligner_ok", "quarantined"]


def parse_wordlist(path: str):
    """Yield (orth, phonemes, syllables_or_None)."""
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[warn] line {ln}: no phoneme column, skipped", file=sys.stderr)
                continue
            orth = unicodedata.normalize("NFC", parts[0].strip())
            syms = parts[1].split()
            if "." in syms:
                syllables, cur = [], []
                for s in syms:
                    if s == ".":
                        if cur:
                            syllables.append(cur)
                        cur = []
                    else:
                        cur.append(s)
                if cur:
                    syllables.append(cur)
                phonemes = [p for syl in syllables for p in syl]
            else:
                syllables, phonemes = None, syms
            yield orth, phonemes, syllables


def compute_rows(entries, tokenizers):
    rows, quarantined = [], []
    for orth, phonemes, syllables in entries:
        aln_ok = align(orth, phonemes).ok
        m = len(segment_aksharas(orth))
        for tk_name, tk in tokenizers.items():
            row = dict.fromkeys(FIELDS)
            row.update(word=orth, tokenizer=tk_name, m_aksharas=m,
                       n_phonemes=len(phonemes), aligner_ok=aln_ok, quarantined=False)
            try:
                spans = real_token_byte_spans(orth, tk)
            except MisalignedTokenizationError:
                row["quarantined"] = True
                quarantined.append((orth, tk_name))
                rows.append(row)
                continue
            b = orth.encode("utf-8")
            row["tokens"] = "|".join(b[s:e].decode("utf-8", "backslashreplace")
                                     for s, e in spans)
            row["n_tokens"] = len(spans)
            g = gtad(orth, spans)
            s = stad_bn(orth, phonemes, spans, syllables=syllables)
            row.update(n_boundaries=g.n_boundaries, gtad=round(g.gtad, 6),
                       gtad_byte=g.byte_internal, gtad_matra=g.matra_split,
                       gtad_conjunct=g.conjunct_split,
                       stad=None if s.stad is None else round(s.stad, 6),
                       rho=round(s.rho, 6))
            rows.append(row)
    return rows, quarantined


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wordlist")
    ap.add_argument("-o", "--out", default="results/metrics.csv")
    ap.add_argument("--tokenizers", nargs="+", default=None,
                    help="subset of: llama3 gpt2 byt5 tigerllm banglat5 (default: all loadable)")
    args = ap.parse_args()

    tokenizers = load_tokenizers(args.tokenizers)
    if not tokenizers:
        sys.exit("no tokenizer could be loaded")

    entries = list(parse_wordlist(args.wordlist))
    print(f"[compute_metrics] {len(entries)} words x {len(tokenizers)} tokenizers")
    rows, quarantined = compute_rows(entries, tokenizers)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_q = len(quarantined)
    print(f"[compute_metrics] wrote {len(rows)} rows -> {out}"
          f"  (quarantined: {n_q})")
    if quarantined:
        for orth, tk in quarantined[:20]:
            print(f"  quarantined: {orth} ({tk})")


if __name__ == "__main__":
    main()

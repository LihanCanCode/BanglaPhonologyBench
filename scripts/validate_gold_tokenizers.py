# -*- coding: utf-8 -*-
"""Validate the real-tokenizer adapter on the 14 gold words and print a
real-vs-simulated comparison table (the SIMULATED tables live in
test_bangla_phonology.py and were illustrative only)."""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from bangla_phonology import gtad, stad_bn, token_byte_spans_from_strings
from gold import GOLD
from src.tokenizer_adapter import (MisalignedTokenizationError,
                                   load_tokenizers, real_token_byte_spans)
from test_bangla_phonology import SIM_TOKENIZERS  # noqa: E402  (root harness)


def metric_str(word, ph, syl, spans):
    g = gtad(word, spans)
    s = stad_bn(word, ph, spans, syllables=syl)
    stad = f"{s.stad:.2f}" if s.stad is not None else " —  "
    return (f"GTAD={g.gtad:.2f}({g.byte_internal}/{g.matra_split}/{g.conjunct_split})"
            f" STAD={stad} rho={s.rho:.2f}")


def main():
    tokenizers = load_tokenizers()
    print()
    for word, ph, _, syl in GOLD:
        w = unicodedata.normalize("NFC", word)
        print(f"=== {w}  /{''.join(p for s_ in syl for p in s_)}/ ===")
        for sim_name, table in SIM_TOKENIZERS.items():
            if w in table:
                spans = token_byte_spans_from_strings(table[w])
                print(f"  {sim_name:<16} {'|'.join(table[w]):<24} "
                      f"{metric_str(w, ph, syl, spans)}   [SIMULATED]")
        for name, tk in tokenizers.items():
            try:
                spans = real_token_byte_spans(w, tk)
            except MisalignedTokenizationError as e:
                print(f"  {name:<16} QUARANTINED: {e}")
                continue
            b = w.encode("utf-8")
            toks = "|".join(b[s:e].decode("utf-8", "backslashreplace") for s, e in spans)
            print(f"  {name:<16} {toks:<24} {metric_str(w, ph, syl, spans)}")
        print()


if __name__ == "__main__":
    main()

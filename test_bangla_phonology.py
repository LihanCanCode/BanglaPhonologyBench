# -*- coding: utf-8 -*-
"""Validation harness: real Bangla words, hand-specified gold (standard colloquial
pronunciations). Tokenizer splits marked SIMULATED are illustrative byte-BPE
behaviors for demonstrating GTAD — real tokenizers plug in via byte spans."""
import unicodedata
from bangla_phonology import (segment_aksharas, syllabify, align, gtad, stad_bn,
                              token_byte_spans_from_strings)

# word, phonemes, expected aksharas, expected syllables
GOLD = [
    ("মানুষ",  ["m","a","n","u","ʃ"],                ["মা","নু","ষ"],      [["m","a"],["n","u","ʃ"]]),
    ("শান্ত",  ["ʃ","a","n","t̪","o"],               ["শা","ন্ত"],         [["ʃ","a","n"],["t̪","o"]]),
    ("কলম",   ["k","ɔ","l","o","m"],               ["ক","ল","ম"],       [["k","ɔ"],["l","o","m"]]),
    ("ঘর",    ["gʱ","ɔ","r"],                      ["ঘ","র"],           [["gʱ","ɔ","r"]]),
    ("বিশ্ব",  ["b","i","ʃ","ʃ","o"],               ["বি","শ্ব"],         [["b","i","ʃ"],["ʃ","o"]]),
    ("অংক",   ["ɔ","ŋ","k","o"],                   ["অং","ক"],          [["ɔ","ŋ"],["k","o"]]),
    ("প্রথম",  ["p","r","o","t̪ʰ","o","m"],          ["প্র","থ","ম"],      [["p","r","o"],["t̪ʰ","o","m"]]),
    ("মন্ত্র",  ["m","o","n","t̪","r","o"],           ["ম","ন্ত্র"],         [["m","o","n"],["t̪","r","o"]]),
    ("চাঁদ",   ["tʃ","ã","d̪"],                      ["চাঁ","দ"],          [["tʃ","ã","d̪"]]),
    ("বিজ্ঞান", ["b","i","g","g","æ","n"],           ["বি","জ্ঞা","ন"],     [["b","i","g"],["g","æ","n"]]),
    ("কৃষ্ণ",  ["k","r","i","ʃ","n","o"],           ["কৃ","ষ্ণ"],         [["k","r","i","ʃ"],["n","o"]]),
    ("স্টেশন", ["s","ʈ","e","ʃ","o","n"],           ["স্টে","শ","ন"],     [["s","ʈ","e"],["ʃ","o","n"]]),
    ("আমার",  ["a","m","a","r"],                   ["আ","মা","র"],      [["a"],["m","a","r"]]),
    ("কাজ",   ["k","a","dʒ"],                      ["কা","জ"],          [["k","a","dʒ"]]),
]

# SIMULATED tokenizations (token strings; adapter converts to byte spans)
SIM_TOKENIZERS = {
    "sim_akshara_bpe": {   # splits only at akshara gaps (best case)
        "শান্ত": ["শা", "ন্ত"], "মানুষ": ["মা", "নুষ"], "মন্ত্র": ["ম", "ন্ত্র"],
        "বিশ্ব": ["বি", "শ্ব"], "প্রথম": ["প্রথ", "ম"], "অংক": ["অং", "ক"],
    },
    "sim_byte_bpe": {      # cluster-breaking splits typical of byte-level BPE
        "শান্ত": ["শ", "া", "ন্ত"],           # matra split
        "মানুষ": ["মান", "ুষ"],              # matra split (boundary before ু)
        "মন্ত্র": ["মন্", "ত্র"],              # conjunct split (after hasanta)
        "বিশ্ব": ["বিশ", "্ব"],               # conjunct split
        "প্রথম": ["প্র", "থম"],               # legal
        "অংক": ["অ", "ংক"],                 # diacritic split
    },
}

def check(name, got, want):
    status = "OK " if got == want else "FAIL"
    print(f"  [{status}] {name}: {got}" + ("" if got == want else f"   expected {want}"))
    return got == want

n_ok = n_all = 0
print("=" * 78)
print("1) AKSHARA SEGMENTATION + PHONEMIC SYLLABIFICATION + ALIGNMENT")
print("=" * 78)
for word, ph, ak_gold, syl_gold in GOLD:
    w = unicodedata.normalize("NFC", word)
    print(f"\n{w}  /{''.join(ph)}/")
    ak = segment_aksharas(w)
    sy = syllabify(ph)
    n_all += 2
    n_ok += check("aksharas ", ak, ak_gold)
    n_ok += check("syllables", sy, syl_gold)
    a = align(w, ph)
    tag = "ok" if a.ok else f"FLAGGED ({a.note})"
    spans = [("".join(ph[s:e]) or "∅") for s, e in a.spans]
    print(f"  align [{tag}]: " + "  ".join(f"{c}→/{p}/" for c, p in zip(ak, spans)))

print(f"\nSegmentation/syllabification: {n_ok}/{n_all} checks passed")

print("\n" + "=" * 78)
print("2) METRICS (GTAD / STAD_bn / rho) on SIMULATED tokenizations")
print("=" * 78)
for tk_name, table in SIM_TOKENIZERS.items():
    print(f"\n--- tokenizer: {tk_name} (SIMULATED) ---")
    print(f"{'word':<10}{'tokens':<28}{'GTAD':>6}{'(byte/matra/conj)':>19}{'STAD':>7}{'rho':>6}")
    for word, ph, _, _ in GOLD:
        if word not in table:
            continue
        toks = table[word]
        spans = token_byte_spans_from_strings(toks)
        g = gtad(word, spans)
        s = stad_bn(word, ph, spans)
        stad_str = f"{s.stad:.2f}" if s.stad is not None else "  — "
        print(f"{word:<10}{'|'.join(toks):<28}{g.gtad:>6.2f}"
              f"{f'({g.byte_internal}/{g.matra_split}/{g.conjunct_split})':>19}"
              f"{stad_str:>7}{s.rho:>6.2f}")

print("\n" + "=" * 78)
print("3) WORKED EXAMPLE — শান্ত (Spec C.5, now computed rather than asserted)")
print("=" * 78)
word, ph = "শান্ত", ["ʃ","a","n","t̪","o"]
for toks in (["শা","ন্ত"], ["শ","া","ন্ত"], ["শান্","ত"]):
    spans = token_byte_spans_from_strings(toks)
    g, s = gtad(word, spans), stad_bn(word, ph, spans)
    print(f"tokens {'|'.join(toks):<14} GTAD={g.gtad:.2f}  "
          f"STAD={s.stad:.2f}  rho={s.rho:.2f}  v_syl={s.v_syl} v_tok={s.v_tok}")
    for d in g.detail:
        print(f"    violation: {d}")

# -*- coding: utf-8 -*-
"""Freeze BanglaPhonologyBench task datasets (Spec A.4 / A.5) from the clean lexicon.

Outputs (data/tasks/):
  g2p.jsonl                 Task 1 — 3,000 words (1,500 HighFreq / 1,500 LowFreq)
  syllable_count_word.jsonl Task 2a — same 3,000 words, gold = lexicon syllable count
  schwa_deletion.jsonl      Task 4 — 1,000 words, binary inherent-vowel vector

Task 3a (rhyme pairs) is NOT built here — see scripts/build_rhyme_dataset.py
and src/rhyme.py, which properly distinguish open/closed final syllables
(Spec A.4 3a) instead of this file's earlier ad hoc rime_of() heuristic.
The canonical Task 3a output is data/task3a_rhyme_pairs.jsonl.

Frequency buckets (Spec A.3.4): zipf from wordfreq(bn) over the lexicon∩wordfreq
pool; HighFreq = top quartile, LowFreq = bottom quartile. etym/pos are null
until the M3 annotation pass. Gold labels come from the lexicon, never from
rule-based G2P (Spec A.1); items are restricted to aligner-confident entries
and coverage is reported.

Deterministic: seed fixed, sorted pools. Usage: python scripts/build_tasks.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import (HASANTA, INDEP_VOWELS, MATRAS, align, gtad,
                              is_nucleus, segment_aksharas, stad_bn)
from src.tokenizer_adapter import (MisalignedTokenizationError,
                                   load_tokenizers, real_token_byte_spans)

SEED = 13
OUT = Path("data/tasks")


def load_clean_lexicon(path="data/lexicon_clean.tsv"):
    """orth -> (syllables, phonemes); aligner-confident entries, first pron wins."""
    lex = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[5] != "True" or cols[0] in lex:
                continue
            syllables = [s.split() for s in cols[1].split(" . ")]
            lex[cols[0]] = syllables
    return lex


def tokenization_block(word, phonemes, syllables, tokenizers):
    out, rho = {}, None
    for name, tk in tokenizers.items():
        try:
            spans = real_token_byte_spans(word, tk)
        except MisalignedTokenizationError:
            out[name] = {"tokens": None, "gtad": None, "stad": None,
                         "quarantined": True}
            continue
        b = word.encode("utf-8")
        g = gtad(word, spans)
        s = stad_bn(word, phonemes, spans, syllables=syllables)
        rho = s.rho
        out[name] = {
            "tokens": [b[x:y].decode("utf-8", "backslashreplace") for x, y in spans],
            "gtad": round(g.gtad, 4),
            "gtad_types": [g.byte_internal, g.matra_split, g.conjunct_split],
            "stad": None if s.stad is None else round(s.stad, 4),
        }
    return out, rho


def base_item(orth, syllables):
    phonemes = [p for s in syllables for p in s]
    return {
        "orth": orth,
        "orth_nfc_codepoints": [f"U+{ord(c):04X}" for c in orth],
        "grapheme_clusters": segment_aksharas(orth),
        "ipa": "".join(phonemes),
        "phonemes": phonemes,
        "syllables_phonemic": syllables,
        "syllable_count": len(syllables),
        "etym": None,          # M3 annotation pass
        "pos": None,           # M3 annotation pass
        "annotation": {"source": "google_lexicon_bn_bd", "verified": False,
                       "annotators": 0},
    }


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items):>5} items -> {path}")


# --- Tasks 1 & 2a -------------------------------------------------------------

def build_g2p_and_syllables(lex, tokenizers, rng):
    import wordfreq
    zipf = {w: wordfreq.zipf_frequency(w, "bn") for w in lex}
    pool = sorted(w for w, z in zipf.items() if z > 0)
    zs = sorted(zipf[w] for w in pool)
    q1, q3 = zs[len(zs) // 4], zs[3 * len(zs) // 4]
    high = sorted(w for w in pool if zipf[w] >= q3)
    low = sorted(w for w in pool if zipf[w] <= q1)
    print(f"[freq] pool={len(pool)}  q1={q1:.2f} q3={q3:.2f}  "
          f"high={len(high)} low={len(low)}")
    words = rng.sample(high, 1500) + rng.sample(low, 1500)
    buckets = ["high"] * 1500 + ["low"] * 1500

    g2p_items, syl_items = [], []
    for k, (w, bucket) in enumerate(zip(words, buckets)):
        syllables = lex[w]
        it = base_item(w, syllables)
        tok, rho = tokenization_block(w, it["phonemes"], syllables, tokenizers)
        it.update(id=f"g2p_{k:05d}", task="g2p", freq_bucket=bucket,
                  zipf=round(zipf[w], 2), rho=rho, tokenization=tok,
                  schwa_vector=None)
        g2p_items.append(it)
        syl_items.append({**it, "id": f"syl_{k:05d}", "task": "syllable_count_word"})
    write_jsonl(OUT / "g2p.jsonl", g2p_items)
    write_jsonl(OUT / "syllable_count_word.jsonl", syl_items)
    return set(words)


# --- Task 4: schwa deletion -----------------------------------------------------

def eligible_positions(orth, syllables):
    """Aksharas that carry the INHERENT vowel (consonant-initial, no matra) —
    the schwa bit is whether that vowel is pronounced (span has a nucleus)."""
    clusters = segment_aksharas(orth)
    phonemes = [p for s in syllables for p in s]
    a = align(orth, phonemes)
    if not a.ok or len(a.spans) != len(clusters):
        return None
    pos, envs, bits = [], [], []
    for gi, cl in enumerate(clusters):
        if cl[0] in INDEP_VOWELS or any(ch in MATRAS for ch in cl):
            continue
        if not any(ch not in (HASANTA,) for ch in cl):
            continue
        s, e = a.spans[gi]
        bit = int(any(is_nucleus(p) for p in phonemes[s:e]))
        env = ("final" if gi == len(clusters) - 1 else
               "post_conjunct" if gi > 0 and HASANTA in clusters[gi - 1] else
               "conjunct" if HASANTA in cl else "medial")
        pos.append(gi); envs.append(env); bits.append(bit)
    return pos, envs, bits


def build_schwa(lex, rng, n=1000):
    by_sig = defaultdict(list)
    for orth in sorted(lex):
        r = eligible_positions(orth, lex[orth])
        if not r or not r[0]:
            continue
        pos, envs, bits = r
        sig = ("final" in envs, "post_conjunct" in envs or "conjunct" in envs)
        by_sig[sig].append((orth, pos, envs, bits))
    # round-robin over strata to over-sample ambiguous environments (Spec A.4.4)
    for v in by_sig.values():
        rng.shuffle(v)
    order = sorted(by_sig, key=lambda s: -len(by_sig[s]))
    picked, i = [], 0
    while len(picked) < n and any(by_sig[s] for s in order):
        s = order[i % len(order)]
        if by_sig[s]:
            picked.append(by_sig[s].pop())
        i += 1
    items = []
    for k, (orth, pos, envs, bits) in enumerate(picked):
        it = base_item(orth, lex[orth])
        it.update(id=f"schwa_{k:05d}", task="schwa_deletion",
                  schwa_positions=pos, schwa_environments=envs,
                  schwa_vector=bits)
        items.append(it)
    write_jsonl(OUT / "schwa_deletion.jsonl", items)


def main():
    rng = random.Random(SEED)
    lex = load_clean_lexicon()
    print(f"[lexicon] aligner-confident unique orths: {len(lex)}")
    tokenizers = load_tokenizers()
    build_g2p_and_syllables(lex, tokenizers, rng)
    build_schwa(lex, rng)


if __name__ == "__main__":
    main()

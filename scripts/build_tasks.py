# -*- coding: utf-8 -*-
"""Freeze BanglaPhonologyBench task datasets (Spec A.4 / A.5) from the clean lexicon.

Outputs (data/tasks/):
  g2p.jsonl                 Task 1 — 3,000 words (1,500 HighFreq / 1,500 LowFreq)
  syllable_count_word.jsonl Task 2a — same 3,000 words, gold = lexicon syllable count
  schwa_deletion.jsonl      Task 4 — 1,000 words, binary inherent-vowel vector
  rhyme_pairs.jsonl         Task 3a — 200 positive + 200 negative pairs
                                      (anti-leakage: no shared final-2-akshara
                                      spelling in positives; orthographic decoys
                                      tagged "hard" in negatives)

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


# --- Task 3a: rhyme pairs -------------------------------------------------------

def rime_of(syllables):
    last = syllables[-1]
    for i, p in enumerate(last):
        if is_nucleus(p):
            return tuple(last[i:])
    return None


def last2_spelling(orth):
    cl = segment_aksharas(orth)
    return "".join(cl[-2:])


def build_rhyme(lex, rng, n_pos=200, n_neg=200):
    import wordfreq
    pool = [w for w in sorted(lex)
            if wordfreq.zipf_frequency(w, "bn") >= 3.0 and len(lex[w]) >= 2]
    by_rime = defaultdict(list)
    for w in pool:
        r = rime_of(lex[w])
        if r:
            by_rime[r].append(w)

    positives, seen = [], set()
    rimes = sorted((r for r, ws in by_rime.items() if len(ws) >= 2),
                   key=lambda r: -len(by_rime[r]))
    candidates = []
    for r in rimes:
        ws = by_rime[r]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                if last2_spelling(a) == last2_spelling(b):
                    continue          # anti-leakage (Liao & Shi): force phonology
                candidates.append((a, b, r))
    rng.shuffle(candidates)
    for a, b, r in candidates:
        if a in seen or b in seen:    # each word appears in at most one pair
            continue
        positives.append((a, b, r))
        seen.update((a, b))
        if len(positives) >= n_pos:
            break

    negatives = []
    # hard subset: same final akshara spelling, different rime (orthographic decoys)
    by_last = defaultdict(list)
    for w in pool:
        by_last[segment_aksharas(w)[-1]].append(w)
    hard = []
    for ws in by_last.values():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if rime_of(lex[ws[i]]) != rime_of(lex[ws[j]]):
                    hard.append((ws[i], ws[j]))
    rng.shuffle(hard)
    negatives += [(a, b, "hard") for a, b in hard[:n_neg // 4]]
    # easy negatives: matched on syllable count, different rime
    flat = pool[:]
    rng.shuffle(flat)
    i = 0
    while len(negatives) < n_neg and i + 1 < len(flat):
        a, b = flat[i], flat[i + 1]
        i += 2
        if (len(lex[a]) == len(lex[b]) and rime_of(lex[a]) != rime_of(lex[b])
                and segment_aksharas(a)[-1] != segment_aksharas(b)[-1]):
            negatives.append((a, b, "easy"))

    items = []
    for k, (a, b, tag) in enumerate([(x, y, "pos") for x, y, _ in positives]
                                    + negatives):
        items.append({
            "id": f"rhyme_{k:05d}", "task": "rhyme_awareness",
            "word1": a, "word2": b,
            "ipa1": "".join(p for s in lex[a] for p in s),
            "ipa2": "".join(p for s in lex[b] for p in s),
            "rime1": list(rime_of(lex[a])), "rime2": list(rime_of(lex[b])),
            "label": 1 if tag == "pos" else 0,
            "subset": tag,
            "annotation": {"source": "google_lexicon_bn_bd", "verified": False},
        })
    write_jsonl(OUT / "rhyme_pairs.jsonl", items)
    print(f"[rhyme] positives={len(positives)} negatives={len(negatives)} "
          f"(hard={sum(1 for *_, t in negatives if t == 'hard')})")


def main():
    rng = random.Random(SEED)
    lex = load_clean_lexicon()
    print(f"[lexicon] aligner-confident unique orths: {len(lex)}")
    tokenizers = load_tokenizers()
    build_g2p_and_syllables(lex, tokenizers, rng)
    build_schwa(lex, rng)
    build_rhyme(lex, rng)


if __name__ == "__main__":
    main()

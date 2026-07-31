# -*- coding: utf-8 -*-
"""Task 3a rhyme-pair dataset builder (Spec A.4 Task 3, A.5 schema).

Reads data/lexicon_clean.tsv (aligner_ok rows only), computes rime()/
assonance_key() (src/rhyme.py) for every entry, and mines:
  - 200 POSITIVE pairs: share a rime key; reject is_trivial_pair, stem-
    sharing (substring), or rime()=None candidates; freq-bucket matched;
    capped at 10 pairs/rime-key; stratified ~50/50 closed- vs open-syllable
    rime type.
  - 200 NEGATIVE pairs, three labeled neg_type subtypes:
      assonance_decoy (~80): same assonance_key, different rime — the hard
        negatives (share the final vowel, nothing else).
      ortho_decoy (~40): same final matra (spelling looks alike — this is
        how the dental/retroflex -তা vs -টা trap is mined) but different
        rime. Deliberately NOT run through is_trivial_pair, which would
        remove exactly the pairs this subtype exists to keep.
      random (~80): different assonance_key entirely; matched on syllable
        count ("length") and freq_bucket to the positives.

`freq_bucket` (Spec A.3.4 style, high/mid/low by wordfreq(bn) zipf tercile)
is stored per-word (freq_bucket1/freq_bucket2, both required to match for
positives and random negatives) rather than one shared field — more
transparent than collapsing two words into a single bucket value.

Every acceptance (positive or negative) is checked against two GLOBAL
registries spanning the whole 400-pair dataset, not just its own subtype:
no word in more than 5 pairs, no duplicate unordered pair. Final sanity
gates re-verify all of this plus positives/negatives == 200/200 before
anything is written, and refuse to write if any gate fails.

Modes:
  --stats     rime-key inventory size, group-size distribution, % of
              lexicon with rime()=None. Run this FIRST.
  (default)   generate, print 20 sample positives + 20 sample negatives
              (grouped by neg_type) for manual native-speaker review.
              DRY RUN — does not write anything.
  --confirm   also write data/task3a_rhyme_pairs.jsonl, but only if every
              sanity gate passes.

Usage:
  python scripts/build_rhyme_dataset.py --stats
  python scripts/build_rhyme_dataset.py
  python scripts/build_rhyme_dataset.py --confirm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import is_nucleus, syllabify
from src.rhyme import assonance_key, final_matra, is_trivial_pair, rime

SEED = 13
LEXICON = Path("data/lexicon_clean.tsv")
OUT = Path("data/task3a_rhyme_pairs.jsonl")

TARGET_POS = 200
TARGET_NEG = {"assonance_decoy": 80, "ortho_decoy": 40, "random": 80}
MAX_PER_RIME_KEY = 10
MAX_PAIRS_PER_WORD = 5


# ----------------------------------------------------------------------------
# Loading and per-word field computation
# ----------------------------------------------------------------------------

def load_lexicon(path=LEXICON):
    entries, seen = [], set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 6:
                continue
            orth, ipa_phonemes, _aksharas, _syllables, _syllable_count, aligner_ok = cols
            if aligner_ok != "True" or orth in seen:
                continue
            seen.add(orth)
            phonemes = [tok for tok in ipa_phonemes.split(" ") if tok != "."]
            entries.append({"orth": orth, "phonemes": phonemes})
    return entries


def freq_bucket(zipf, q1, q3):
    if zipf <= 0:
        return None
    if zipf >= q3:
        return "high"
    if zipf <= q1:
        return "low"
    return "mid"


def annotate(entries):
    import wordfreq
    zipfs = [wordfreq.zipf_frequency(e["orth"], "bn") for e in entries]
    known = sorted(z for z in zipfs if z > 0)
    q1 = known[len(known) // 4] if known else 0.0
    q3 = known[3 * len(known) // 4] if known else 0.0
    for e, z in zip(entries, zipfs):
        e["rime"] = rime(e["orth"], e["phonemes"])
        e["assonance_key"] = assonance_key(e["orth"], e["phonemes"])
        e["final_matra"] = final_matra(e["orth"])
        e["syll_count"] = len(syllabify(e["phonemes"]))
        e["zipf"] = z
        e["bucket"] = freq_bucket(z, q1, q3)
    return entries, (q1, q3)


def classify_rime_type(rime_key):
    """'closed' if the rime starts with the nucleus (nucleus+coda), else
    'open' (onset+...+nucleus, nucleus last)."""
    return "closed" if is_nucleus(rime_key[0]) else "open"


# ----------------------------------------------------------------------------
# --stats
# ----------------------------------------------------------------------------

def run_stats(entries):
    total = len(entries)
    none_reason = Counter()
    for e in entries:
        if e["rime"] is not None:
            continue
        syllables = syllabify(e["phonemes"])
        if not syllables:
            none_reason["no_syllables"] += 1
            continue
        final = syllables[-1]
        nucleus_idxs = [i for i, p in enumerate(final) if is_nucleus(p)]
        if not nucleus_idxs:
            none_reason["no_nucleus_in_final_syllable"] += 1
        else:
            none_reason["open_syllable_no_onset"] += 1

    n_none = sum(none_reason.values())
    groups = defaultdict(list)
    for e in entries:
        if e["rime"] is not None:
            groups[e["rime"]].append(e["orth"])

    sizes = Counter()
    for g in groups.values():
        n = len(g)
        if n == 1:
            sizes["1 (singleton, unusable for pairing)"] += 1
        elif n == 2:
            sizes["2"] += 1
        elif n <= 5:
            sizes["3-5"] += 1
        elif n <= 10:
            sizes["6-10"] += 1
        elif n <= 20:
            sizes["11-20"] += 1
        else:
            sizes["21+"] += 1

    type_counts = Counter(classify_rime_type(k) for k in groups)
    pairable_words_by_type = Counter()
    for k, members in groups.items():
        if len(members) >= 2:
            pairable_words_by_type[classify_rime_type(k)] += len(members)

    n_known_freq = sum(1 for e in entries if e["bucket"] is not None)

    print("=" * 70)
    print("TASK 3a RHYME DATASET — --stats")
    print("=" * 70)
    print(f"aligner_ok lexicon entries scanned : {total}")
    print(f"rime() == None                     : {n_none}  ({n_none/total:.1%})")
    for reason, n in none_reason.most_common():
        print(f"    {reason:<32}: {n}  ({n/total:.1%})")
    print(f"distinct non-None rime keys         : {len(groups)}")
    print(f"  of which 'closed' type            : {type_counts['closed']}")
    print(f"  of which 'open' type               : {type_counts['open']}")
    print(f"group-size distribution:")
    for label in ("1 (singleton, unusable for pairing)", "2", "3-5", "6-10", "11-20", "21+"):
        if sizes[label]:
            print(f"    {label:<38}: {sizes[label]}")
    print(f"words in groups with >=2 members (pairable):")
    print(f"    closed-type                       : {pairable_words_by_type['closed']}")
    print(f"    open-type                          : {pairable_words_by_type['open']}")
    print(f"words with known wordfreq(bn) zipf   : {n_known_freq}/{total} ({n_known_freq/total:.1%})")
    print()
    max_pairs_closed = sum(min(len(m), MAX_PER_RIME_KEY) * (len(m) - 1) // 2
                           for k, m in groups.items() if len(m) >= 2 and classify_rime_type(k) == "closed")
    max_pairs_open = sum(min(len(m), MAX_PER_RIME_KEY) * (len(m) - 1) // 2
                         for k, m in groups.items() if len(m) >= 2 and classify_rime_type(k) == "open")
    print(f"upper bound on distinct positive pairs (post rime-key cap of {MAX_PER_RIME_KEY}, "
          f"BEFORE trivial-pair/stem/freq filters):")
    print(f"    closed-type                       : {max_pairs_closed}")
    print(f"    open-type                          : {max_pairs_open}")
    print(f"    total                             : {max_pairs_closed + max_pairs_open}  "
          f"(target: {TARGET_POS}, {TARGET_POS//2}/{TARGET_POS//2} split)")


# ----------------------------------------------------------------------------
# Global registries shared across ALL pairs (positives + every neg_type)
# ----------------------------------------------------------------------------

class Registry:
    def __init__(self):
        self.word_count = Counter()
        self.used_pairs = set()

    def can_use(self, w1, w2):
        if w1 == w2:
            return False
        key = tuple(sorted((w1, w2)))
        if key in self.used_pairs:
            return False
        if self.word_count[w1] >= MAX_PAIRS_PER_WORD or self.word_count[w2] >= MAX_PAIRS_PER_WORD:
            return False
        return True

    def commit(self, w1, w2):
        self.word_count[w1] += 1
        self.word_count[w2] += 1
        self.used_pairs.add(tuple(sorted((w1, w2))))


MIN_STEM_PHONEME_LEN = 3   # avoid spurious matches on short/common phoneme runs


def _is_contig_sublist(short, long_):
    n = len(short)
    if n < MIN_STEM_PHONEME_LEN or n > len(long_):
        return False
    return any(long_[i:i + n] == short for i in range(len(long_) - n + 1))


def is_stem_pair(e1, e2):
    """True if the words share a stem — orthographically (one spelling is a
    substring of the other) OR phonemically (one's pronunciation is a
    contiguous run inside the other's, e.g. খেত /kʰet̪/ inside নীলক্ষেত
    /nilkʰet̪/ "blue-field": same morpheme, spelled with the conjunct ক্ষ
    instead of plain খ, so the orthographic check alone misses it)."""
    w1, w2 = e1["orth"], e2["orth"]
    if w1 in w2 or w2 in w1:
        return True
    ph1, ph2 = e1["phonemes"], e2["phonemes"]
    shorter, longer = (ph1, ph2) if len(ph1) <= len(ph2) else (ph2, ph1)
    return _is_contig_sublist(shorter, longer)


# ----------------------------------------------------------------------------
# Positives
# ----------------------------------------------------------------------------

def build_positives(entries, reg, rng, target=TARGET_POS):
    groups = defaultdict(list)
    for e in entries:
        if e["rime"] is not None:
            groups[e["rime"]].append(e)

    by_type = {"closed": [], "open": []}
    keys = list(groups.keys())
    rng.shuffle(keys)

    per_type_target = target // 2
    for key in keys:
        rtype = classify_rime_type(key)
        if len(by_type[rtype]) >= per_type_target:
            continue
        members = groups[key]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        pairs = list(combinations(members, 2))
        rng.shuffle(pairs)
        n_taken = 0
        for e1, e2 in pairs:
            if n_taken >= MAX_PER_RIME_KEY or len(by_type[rtype]) >= per_type_target:
                break
            w1, w2 = e1["orth"], e2["orth"]
            if not reg.can_use(w1, w2):
                continue
            if e1["phonemes"] == e2["phonemes"]:
                continue                       # exact homophone/spelling variant, not a rhyme
            if is_trivial_pair(w1, w2) or is_stem_pair(e1, e2):
                continue
            if e1["bucket"] is None or e1["bucket"] != e2["bucket"]:
                continue
            reg.commit(w1, w2)
            n_taken += 1
            by_type[rtype].append({
                "orth1": w1, "orth2": w2,
                "phonemes1": e1["phonemes"], "phonemes2": e2["phonemes"],
                "rime1": list(key), "rime2": list(key),
                "assonance_key1": e1["assonance_key"], "assonance_key2": e2["assonance_key"],
                "neg_type": None, "freq_bucket1": e1["bucket"], "freq_bucket2": e2["bucket"],
                "label": 1,
            })
    return by_type["closed"] + by_type["open"], {k: len(v) for k, v in by_type.items()}


# ----------------------------------------------------------------------------
# Negatives
# ----------------------------------------------------------------------------

def build_assonance_decoys(entries, reg, rng, target):
    groups = defaultdict(list)
    for e in entries:
        if e["rime"] is not None and e["assonance_key"] is not None:
            groups[e["assonance_key"]].append(e)

    out = []
    keys = list(groups.keys())
    rng.shuffle(keys)
    for key in keys:
        if len(out) >= target:
            break
        members = groups[key]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        pairs = list(combinations(members, 2))
        rng.shuffle(pairs)
        for e1, e2 in pairs:
            if len(out) >= target:
                break
            if e1["rime"] == e2["rime"]:
                continue                       # would actually rhyme
            w1, w2 = e1["orth"], e2["orth"]
            if not reg.can_use(w1, w2):
                continue
            if is_trivial_pair(w1, w2) or is_stem_pair(e1, e2):
                continue
            reg.commit(w1, w2)
            out.append({
                "orth1": w1, "orth2": w2,
                "phonemes1": e1["phonemes"], "phonemes2": e2["phonemes"],
                "rime1": list(e1["rime"]), "rime2": list(e2["rime"]),
                "assonance_key1": e1["assonance_key"], "assonance_key2": e2["assonance_key"],
                "neg_type": "assonance_decoy",
                "freq_bucket1": e1["bucket"], "freq_bucket2": e2["bucket"],
                "label": 0,
            })
    return out


def build_ortho_decoys(entries, reg, rng, target):
    groups = defaultdict(list)
    for e in entries:
        if e["rime"] is not None and e["final_matra"] is not None:
            groups[e["final_matra"]].append(e)

    out = []
    keys = list(groups.keys())
    rng.shuffle(keys)
    for key in keys:
        if len(out) >= target:
            break
        members = groups[key]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        pairs = list(combinations(members, 2))
        rng.shuffle(pairs)
        for e1, e2 in pairs:
            if len(out) >= target:
                break
            if e1["rime"] == e2["rime"]:
                continue
            w1, w2 = e1["orth"], e2["orth"]
            if not reg.can_use(w1, w2):
                continue
            if is_stem_pair(e1, e2):
                continue
            # deliberately NOT filtering is_trivial_pair here — that would
            # remove exactly the same-looking-spelling pairs this subtype
            # exists to keep (Spec: "orthographic decoys" hard negatives)
            reg.commit(w1, w2)
            out.append({
                "orth1": w1, "orth2": w2,
                "phonemes1": e1["phonemes"], "phonemes2": e2["phonemes"],
                "rime1": list(e1["rime"]), "rime2": list(e2["rime"]),
                "assonance_key1": e1["assonance_key"], "assonance_key2": e2["assonance_key"],
                "neg_type": "ortho_decoy",
                "freq_bucket1": e1["bucket"], "freq_bucket2": e2["bucket"],
                "label": 0,
            })
    return out


def build_random_negatives(entries, reg, rng, target):
    pools = defaultdict(list)
    for e in entries:
        if e["assonance_key"] is not None and e["bucket"] is not None:
            pools[(e["syll_count"], e["bucket"])].append(e)

    out = []
    pool_keys = list(pools.keys())
    rng.shuffle(pool_keys)
    for pk in pool_keys:
        if len(out) >= target:
            break
        members = pools[pk]
        if len(members) < 2:
            continue
        rng.shuffle(members)
        pairs = list(combinations(members, 2))
        rng.shuffle(pairs)
        for e1, e2 in pairs:
            if len(out) >= target:
                break
            if e1["assonance_key"] == e2["assonance_key"]:
                continue                       # need a real vowel mismatch
            w1, w2 = e1["orth"], e2["orth"]
            if not reg.can_use(w1, w2):
                continue
            if is_trivial_pair(w1, w2) or is_stem_pair(e1, e2):
                continue
            reg.commit(w1, w2)
            out.append({
                "orth1": w1, "orth2": w2,
                "phonemes1": e1["phonemes"], "phonemes2": e2["phonemes"],
                "rime1": list(e1["rime"]) if e1["rime"] else None,
                "rime2": list(e2["rime"]) if e2["rime"] else None,
                "assonance_key1": e1["assonance_key"], "assonance_key2": e2["assonance_key"],
                "neg_type": "random",
                "freq_bucket1": e1["bucket"], "freq_bucket2": e2["bucket"],
                "label": 0,
            })
    return out


# ----------------------------------------------------------------------------
# Sanity gates
# ----------------------------------------------------------------------------

def run_sanity_gates(positives, negatives):
    errors = []
    all_pairs = positives + negatives

    seen_pairs = set()
    for r in all_pairs:
        key = tuple(sorted((r["orth1"], r["orth2"])))
        if key in seen_pairs:
            errors.append(f"duplicate pair: {key}")
        seen_pairs.add(key)

    word_count = Counter()
    for r in all_pairs:
        word_count[r["orth1"]] += 1
        word_count[r["orth2"]] += 1
    overused = {w: n for w, n in word_count.items() if n > MAX_PAIRS_PER_WORD}
    if overused:
        errors.append(f"words exceeding {MAX_PAIRS_PER_WORD} pairs: {overused}")

    if len(positives) != TARGET_POS:
        errors.append(f"positives = {len(positives)}, expected {TARGET_POS}")
    neg_by_type = Counter(r["neg_type"] for r in negatives)
    if len(negatives) != TARGET_NEG_TOTAL:
        errors.append(f"negatives = {len(negatives)}, expected {TARGET_NEG_TOTAL} "
                      f"(by type: {dict(neg_by_type)})")

    for r in positives:
        if r["rime1"] != r["rime2"]:
            errors.append(f"positive pair with mismatched rime: {r['orth1']}/{r['orth2']}")
    for r in negatives:
        if r["rime1"] == r["rime2"] and r["rime1"] is not None:
            errors.append(f"negative pair ({r['neg_type']}) with MATCHING rime: "
                          f"{r['orth1']}/{r['orth2']}")

    return errors


TARGET_NEG_TOTAL = sum(TARGET_NEG.values())


# ----------------------------------------------------------------------------
# Review printout
# ----------------------------------------------------------------------------

def print_sample(positives, negatives, rng, n_pos=20):
    print("\n" + "=" * 70)
    print(f"SAMPLE: {n_pos} random POSITIVES (of {len(positives)})")
    print("=" * 70)
    for r in rng.sample(positives, min(n_pos, len(positives))):
        print(f"  {r['orth1']:<14} /{' '.join(r['phonemes1'])}/  <->  "
              f"{r['orth2']:<14} /{' '.join(r['phonemes2'])}/   "
              f"rime={''.join(r['rime1'])}  bucket={r['freq_bucket1']}/{r['freq_bucket2']}")

    by_type = defaultdict(list)
    for r in negatives:
        by_type[r["neg_type"]].append(r)
    quota = {"assonance_decoy": 8, "ortho_decoy": 4, "random": 8}
    print("\n" + "=" * 70)
    print(f"SAMPLE: 20 random NEGATIVES grouped by neg_type "
          f"(of {len(negatives)}: {dict(Counter(r['neg_type'] for r in negatives))})")
    print("=" * 70)
    for ntype, q in quota.items():
        pool = by_type[ntype]
        print(f"\n--- {ntype} ({len(pool)} total) ---")
        for r in rng.sample(pool, min(q, len(pool))):
            r1 = "".join(r["rime1"]) if r["rime1"] else "None"
            r2 = "".join(r["rime2"]) if r["rime2"] else "None"
            print(f"  {r['orth1']:<14} /{' '.join(r['phonemes1'])}/  rime={r1:<8}  vs  "
                  f"{r['orth2']:<14} /{' '.join(r['phonemes2'])}/  rime={r2:<8}  "
                  f"assonance={r['assonance_key1']}/{r['assonance_key2']}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--confirm", action="store_true",
                    help="actually write the output file (after sanity gates pass)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    entries = load_lexicon()
    entries, (q1, q3) = annotate(entries)
    print(f"[load] {len(entries)} aligner_ok lexicon entries "
          f"(zipf tercile thresholds: q1={q1:.2f} q3={q3:.2f})")

    if args.stats:
        run_stats(entries)
        return

    rng = random.Random(args.seed)
    reg = Registry()

    positives, pos_type_counts = build_positives(entries, reg, rng)
    print(f"[positives] {len(positives)}/{TARGET_POS}  (by rime type: {pos_type_counts})")

    assonance = build_assonance_decoys(entries, reg, rng, TARGET_NEG["assonance_decoy"])
    print(f"[assonance_decoy] {len(assonance)}/{TARGET_NEG['assonance_decoy']}")
    ortho = build_ortho_decoys(entries, reg, rng, TARGET_NEG["ortho_decoy"])
    print(f"[ortho_decoy] {len(ortho)}/{TARGET_NEG['ortho_decoy']}")
    rand = build_random_negatives(entries, reg, rng, TARGET_NEG["random"])
    print(f"[random] {len(rand)}/{TARGET_NEG['random']}")
    negatives = assonance + ortho + rand

    print_sample(positives, negatives, rng)

    errors = run_sanity_gates(positives, negatives)
    print("\n" + "=" * 70)
    if errors:
        print(f"SANITY GATES: FAILED ({len(errors)} issue(s))")
        for e in errors[:30]:
            print(f"  - {e}")
        print("=" * 70)
        if args.confirm:
            sys.exit("Refusing to write: sanity gates failed. See above.")
        return
    print("SANITY GATES: all passed "
          f"(positives=200, negatives=200, no duplicate pairs, "
          f"no word in >{MAX_PAIRS_PER_WORD} pairs)")
    print("=" * 70)

    if not args.confirm:
        print("\nDRY RUN — review the sample above. Re-run with --confirm to write "
              f"{OUT}.")
        return

    records = []
    for k, r in enumerate(positives + negatives):
        records.append({"id": f"rhyme3a_{k:05d}", "task": "rhyme_awareness", **r})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} pairs -> {OUT}")


if __name__ == "__main__":
    main()

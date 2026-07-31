# -*- coding: utf-8 -*-
"""Task 3b rhyme-generation dataset builder (Spec A.4 3b, A.5 schema).

Unlike 3a (classification: does this pair rhyme?), 3b is generation: given
a prompt word, produce rhyming words; gold = every other aligner_ok lexicon
word sharing its rime key (src/rhyme.py's rime(), same open/closed-syllable
rule as 3a — no separate logic here, reuses build_rhyme_dataset.py's
loading/annotation and src/rhyme.py's rime()).

DEVIATION FROM THE LITERAL SPEC TEXT ("gold = all lexicon words with
matching rime"): gold sets are filtered through the SAME anti-leakage
check as 3a (src/rhyme.py's is_trivial_pair, via the precomputed-key fast
path — a naive O(group_size^2) pairwise pass is too slow for a 7,000+
member rime group). Reason: short grammatical suffixes (-এর genitive, -রা
plural) occupy the ENTIRE "final 2 akshara" window is_trivial_pair checks,
so essentially every word carrying that suffix collides on it regardless
of root -- verified empirically before adopting this: একের's raw gold
was 7,127 words (basically "any noun + -এর"), 20 after filtering; a
genuine derivational-suffix rime like -তা (ভাতা) barely moves (639 -> 637)
since the akshara before -তা varies by root. Filtering out pure-suffix
matches makes Success Rate@5 measure real phonological rhyme generation
rather than "can you append -এর to a real word," at the cost of diverging
from the spec's literal wording. Approved deviation, see BanglaPhonology
Bench_Research_Spec.md's Task 3b note.

300 prompts: 200 common (top zipf tercile) + 100 rare (bottom zipf tercile),
each required to have a non-None rime AND at least MIN_GOLD FILTERED gold
words (an unanswerable prompt with an empty gold set is not a useful eval
item). Gold sets are otherwise NOT capped — Success Rate@k (src/rhyme.py
success_at_k / mean_success_at_k) only needs set membership, so a large
gold set costs nothing at eval time.

NOT implemented here: Spec A.4 3b also says "mine additional attested
rhymes from poetry corpora to enrich gold" (Tagore/Nazrul, per Spec A.2).
No poetry corpus is present in this repo — gold sets here are lexicon-only.
This under-counts true gold (an attested poetic rhyme not sharing an exact
phonemic rime, e.g. via historical/dialectal pronunciation, would be missed
and could make a correct model generation register as a miss), so Success
Rate@k computed against this dataset is a lower bound. Flagged, not solved.

Modes:
  --stats     rime-key/gold-set-size distribution for the sampling pool —
              run first to confirm 300 usable prompts exist at a sane
              minimum gold-set-size threshold.
  (default)   generate, print a review sample. DRY RUN — does not write.
  --confirm   also write data/task3b_rhyme_generation.jsonl.

Usage:
  python scripts/build_rhyme_generation_dataset.py --stats
  python scripts/build_rhyme_generation_dataset.py
  python scripts/build_rhyme_generation_dataset.py --confirm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_rhyme_dataset import annotate, load_lexicon  # noqa: E402
from src.rhyme import is_trivial_pair_fast, trivial_pair_key  # noqa: E402

SEED = 13
OUT = Path("data/task3b_rhyme_generation.jsonl")

TARGET_COMMON = 200
TARGET_RARE = 100
MIN_GOLD = 3   # a prompt needs at least this many OTHER lexicon rhymes to be usable


def build_rime_groups(entries):
    """rime -> list of (orth, trivial_pair_key) — the key is precomputed
    ONCE per word here, not per pair, since a prompt gets checked against
    every other member of its (sometimes 1000s-large) rime group."""
    groups = defaultdict(list)
    for e in entries:
        if e["rime"] is not None:
            groups[e["rime"]].append((e["orth"], trivial_pair_key(e["orth"])))
    return groups


def filtered_gold_words(prompt_orth, prompt_key, members):
    """members' orths minus the prompt itself and anything is_trivial_pair
    with it (see module docstring for why this filter is applied here)."""
    return [orth for orth, key in members
            if orth != prompt_orth and not is_trivial_pair_fast(prompt_key, key)]


def run_stats(entries):
    groups = build_rime_groups(entries)
    gold_sizes = Counter()
    for members in groups.values():
        for orth, key in members:
            n = len(filtered_gold_words(orth, key, members))
            if n == 0:
                gold_sizes["0 (unusable)"] += 1
            elif n < 3:
                gold_sizes["1-2"] += 1
            elif n < 10:
                gold_sizes["3-9"] += 1
            elif n < 30:
                gold_sizes["10-29"] += 1
            else:
                gold_sizes["30+"] += 1

    usable_words = sum(1 for members in groups.values() for orth, key in members
                       if len(filtered_gold_words(orth, key, members)) >= MIN_GOLD)
    n_known_freq = sum(1 for e in entries if e["bucket"] is not None)

    print("=" * 70)
    print("TASK 3b RHYME GENERATION — --stats")
    print("=" * 70)
    print(f"aligner_ok lexicon entries scanned : {len(entries)}")
    print(f"distinct non-None rime keys         : {len(groups)}")
    print(f"post-filter gold-set-size distribution (one bucket per WORD, "
          f"as if it were the prompt):")
    for label in ("0 (unusable)", "1-2", "3-9", "10-29", "30+"):
        if gold_sizes[label]:
            print(f"    {label:<16}: {gold_sizes[label]} words")
    print(f"words usable as a prompt (gold >= {MIN_GOLD})  : {usable_words}"
          f"  (target: {TARGET_COMMON + TARGET_RARE})")
    print(f"words with known wordfreq(bn) zipf   : {n_known_freq}/{len(entries)}")


def build_prompts(entries, rng):
    groups = build_rime_groups(entries)
    key_of = {orth: key for members in groups.values() for orth, key in members}

    pool = [e for e in entries if e["bucket"] is not None and e["rime"] is not None]
    common_pool = sorted((e for e in pool if e["bucket"] == "high"), key=lambda e: e["orth"])
    rare_pool = sorted((e for e in pool if e["bucket"] == "low"), key=lambda e: e["orth"])
    rng.shuffle(common_pool)
    rng.shuffle(rare_pool)

    def take(pool, n, tag):
        out = []
        for e in pool:
            if len(out) >= n:
                break
            members = groups[e["rime"]]
            gold = filtered_gold_words(e["orth"], key_of[e["orth"]], members)
            if len(gold) < MIN_GOLD:
                continue                       # not enough real rhymes after anti-leakage
            out.append({
                "prompt_word": e["orth"], "prompt_phonemes": e["phonemes"],
                "rime": list(e["rime"]), "freq_tag": tag,
                "gold_rhymes": gold, "gold_count": len(gold),
            })
        return out

    common = take(common_pool, TARGET_COMMON, "common")
    rare = take(rare_pool, TARGET_RARE, "rare")
    return common, rare


def print_sample(common, rare, rng, n=10):
    print("\n" + "=" * 70)
    print(f"SAMPLE: {n} random COMMON prompts (of {len(common)})")
    print("=" * 70)
    for r in rng.sample(common, min(n, len(common))):
        gold_preview = ", ".join(r["gold_rhymes"][:6])
        more = f" (+{r['gold_count']-6} more)" if r["gold_count"] > 6 else ""
        print(f"  {r['prompt_word']:<16} /{' '.join(r['prompt_phonemes'])}/  "
              f"rime={''.join(r['rime']):<8}  gold[{r['gold_count']}]: {gold_preview}{more}")

    print("\n" + "=" * 70)
    print(f"SAMPLE: {n} random RARE prompts (of {len(rare)})")
    print("=" * 70)
    for r in rng.sample(rare, min(n, len(rare))):
        gold_preview = ", ".join(r["gold_rhymes"][:6])
        more = f" (+{r['gold_count']-6} more)" if r["gold_count"] > 6 else ""
        print(f"  {r['prompt_word']:<16} /{' '.join(r['prompt_phonemes'])}/  "
              f"rime={''.join(r['rime']):<8}  gold[{r['gold_count']}]: {gold_preview}{more}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--confirm", action="store_true")
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
    common, rare = build_prompts(entries, rng)
    print(f"[prompts] common={len(common)}/{TARGET_COMMON}  rare={len(rare)}/{TARGET_RARE}")

    if len(common) < TARGET_COMMON or len(rare) < TARGET_RARE:
        print(f"WARNING: shortfall against target (300 total). "
              f"Got {len(common) + len(rare)}.")

    print_sample(common, rare, rng)

    gold_counts = [r["gold_count"] for r in common + rare]
    print(f"\ngold-set size across the 300 prompts: "
          f"min={min(gold_counts)} median={sorted(gold_counts)[len(gold_counts)//2]} "
          f"max={max(gold_counts)}")

    if not args.confirm:
        print(f"\nDRY RUN — review the sample above. Re-run with --confirm to write {OUT}.")
        return

    records = []
    for k, r in enumerate(common + rare):
        records.append({
            "id": f"rhyme3b_{k:05d}", "task": "rhyme_generation", **r,
            "annotation": {"source": "google_lexicon_bn_bd", "verified": False, "annotators": 0},
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} prompts -> {OUT}")


if __name__ == "__main__":
    main()

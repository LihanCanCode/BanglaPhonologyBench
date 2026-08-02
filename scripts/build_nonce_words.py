# -*- coding: utf-8 -*-
"""M5 contamination check (see docs/DEVELOPMENT_LOG.md, Finding 7b /
"rho-ceiling" section): syllable_count's zero-shot 100% exact-match result
is suspiciously perfect. To tell genuine rule-learning apart from
memorization of this benchmark's specific (public, lexicon-derived) word
list, generate words TigerLLM cannot have seen verbatim in any training
corpus, with gold syllable counts that don't require human annotation.

Method: compound 2-3 real, aligner-confident lexicon words together
(e.g. প্রতি + স্বর + মাপক -> প্রতিস্বরমাপক), the same way real Bangla
compounds are formed, then re-syllabify the CONCATENATED phoneme sequence
with the validated `syllabify()` (not a naive sum of the components' own
counts, so any onset-maximization interaction exactly at the new
component boundary is still handled correctly). This is NOT a full
phonotactic-grammar nonce-word generator (that would need a generative
model of licit Bangla root/affix structure this project doesn't have) --
it's compositional novelty: every syllable/akshara is real and attested,
but the specific multi-morpheme string is (checked) absent from the
65k-entry lexicon and from this benchmark's own frozen word lists, so it
cannot be a verbatim training-set/benchmark-set match.

Known simplification, stated not hidden: no cross-morpheme sandhi
modeling (vowel elision/consonant assimilation at compound boundaries,
which real Bangla compounding sometimes has) -- the gold syllable count
is exactly what `syllabify()` computes on the naive phoneme concatenation.
This slightly overestimates syllable count for the (a minority of)
combinations where real sandhi would apply, a limitation worth stating
alongside any result from this test, not a fatal flaw for the specific
question being asked (does the model's accuracy survive novel wordforms,
or does it collapse).

Run: `python -X utf8 scripts/build_nonce_words.py --confirm -n 150`
(dry-run without --confirm: prints a sample + stats, writes nothing)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_phonology import normalize_bn, syllabify

REPO_ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = REPO_ROOT / "data" / "lexicon_clean.tsv"
OUT_PATH = REPO_ROOT / "data" / "tasks" / "nonce_syllable_count.jsonl"

SEED = 17
MIN_COMPONENT_SYLLABLES = 2   # avoid single-akshara components (too generic,
                              # e.g. bare case markers) skewing every compound
MAX_COMPONENT_SYLLABLES = 4


def load_lexicon():
    """(orth, phonemes) for aligner_ok entries, phonemes as a flat list
    (the '.' syllable-boundary markers in the TSV are dropped -- we
    re-syllabify from scratch at compound-build time anyway)."""
    entries = []
    all_orths = set()
    with open(LEXICON_PATH, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            orth = normalize_bn(parts[idx["orth"]])
            all_orths.add(orth)
            if parts[idx["aligner_ok"]] != "True":
                continue
            n_syl = int(parts[idx["syllable_count"]])
            if not (MIN_COMPONENT_SYLLABLES <= n_syl <= MAX_COMPONENT_SYLLABLES):
                continue
            phonemes = [p for p in parts[idx["ipa_phonemes"]].split() if p != "."]
            entries.append((orth, phonemes))
    return entries, all_orths


def build_nonce_words(entries, all_orths, n, rng):
    seen = set()
    out = []
    attempts = 0
    max_attempts = n * 50
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        k = rng.choice([2, 2, 3])   # mostly 2-word compounds, some 3-word
        parts = rng.sample(entries, k)
        orth = "".join(p[0] for p in parts)
        if orth in seen or orth in all_orths:
            continue
        phonemes = [ph for p in parts for ph in p[1]]
        syllables = syllabify(phonemes)
        n_syl = len(syllables)
        if n_syl < 2:
            continue
        seen.add(orth)
        out.append({
            "id": f"nonce_{len(out):05d}",
            "task": "nonce_syllable_count",
            "orth": orth,
            "phonemes": phonemes,
            "syllables_phonemic": syllables,
            "syllable_count": n_syl,
            "components": [p[0] for p in parts],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    entries, all_orths = load_lexicon()
    print(f"[nonce] {len(entries)} aligner-confident components "
          f"({MIN_COMPONENT_SYLLABLES}-{MAX_COMPONENT_SYLLABLES} syllables) available")

    rng = random.Random(SEED)
    items = build_nonce_words(entries, all_orths, args.n, rng)
    print(f"[nonce] built {len(items)}/{args.n} requested nonce words "
          f"(all verified absent from the {len(all_orths)}-entry lexicon)")

    dist = {}
    for it in items:
        dist[it["syllable_count"]] = dist.get(it["syllable_count"], 0) + 1
    print("[nonce] syllable-count distribution:", dict(sorted(dist.items())))

    print("\n[nonce] sample:")
    for it in rng.sample(items, min(10, len(items))):
        print(f"  {it['orth']:20s} ({' + '.join(it['components'])}) "
              f"-> {it['syllable_count']} syllables")

    if not args.confirm:
        print("\n[nonce] dry run -- pass --confirm to write "
              f"{OUT_PATH.relative_to(REPO_ROOT)}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n[nonce] wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

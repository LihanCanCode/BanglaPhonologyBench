# -*- coding: utf-8 -*-
"""Rhyme primitives for BanglaPhonologyBench Task 3 (Spec A.4 Task 3).

Built entirely on the validated reference implementation — segmentation,
syllabification, and alignment are never reimplemented here, only composed:
`syllabify()` for the phonemic-form rime/assonance keys, `segment_aksharas()`
for the orthographic anti-leakage filter.

Rhyme is computed on the PHONEMIC form only. Two regimes for the final
syllable (Spec A.4 3a):
  - CLOSED (has a coda): rime = nucleus + coda.
  - OPEN (ends in the nucleus, no coda): rime = onset + nucleus — matching
    only the vowel would be assonance (স্বরান্ত্যমিল), not rhyme (অন্ত্যমিল);
    Bangla poetic convention requires the preceding consonant too. An
    onset-less open final syllable (word ends in a bare vowel with nothing
    before it) has no rime at all — any match would be pure assonance — and
    returns None.

Phoneme identity is strict throughout: dental t̪/d̪ vs retroflex ʈ/ɖ,
aspirated vs unaspirated, and nasalized vs oral vowels are all distinct
phoneme strings already in this project's IPA convention, so ordinary tuple/
string equality already enforces this — no separate "strict mode" needed.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from bangla_phonology import MATRAS, is_nucleus, normalize_bn, segment_aksharas, syllabify

__all__ = ["rime", "assonance_key", "is_trivial_pair", "trivial_pair_key",
          "is_trivial_pair_fast", "final_matra", "success_at_k", "mean_success_at_k"]


def _final_syllable_nucleus_index(phonemes: List[str]) -> Optional[Tuple[List[str], int]]:
    """(final_syllable, nucleus_index) for the word's phonemic form, or None
    if there is no syllable / no nucleus in it (shouldn't happen on
    aligner-confident lexicon entries, but words are not guaranteed-valid
    inputs from arbitrary callers)."""
    syllables = syllabify(phonemes)
    if not syllables:
        return None
    final = syllables[-1]
    nucleus_idxs = [i for i, p in enumerate(final) if is_nucleus(p)]
    if not nucleus_idxs:
        return None
    return final, nucleus_idxs[-1]


def rime(word: str, phonemes: List[str]) -> Optional[Tuple[str, ...]]:
    """The rhyme key: nucleus+coda if the final syllable is closed, else
    onset+nucleus if open (returns None for an onset-less open syllable —
    unusable, pure assonance). `word` is accepted for API symmetry/future
    orthographic tie-breaks but the computation is phonemic-only."""
    res = _final_syllable_nucleus_index(phonemes)
    if res is None:
        return None
    final, ni = res
    if ni == len(final) - 1:                  # OPEN: nucleus is syllable-final
        onset = final[:ni]
        if not onset:
            return None                        # vowel-only syllable: assonance only
        return tuple(final[:ni + 1])
    return tuple(final[ni:])                   # CLOSED: nucleus + coda


def assonance_key(word: str, phonemes: List[str]) -> Optional[str]:
    """Just the final syllable's nucleus phoneme — used for hard negatives
    (assonance without rhyme), never as a positive-pair criterion."""
    res = _final_syllable_nucleus_index(phonemes)
    if res is None:
        return None
    final, ni = res
    return final[ni]


# Common inflectional/classifier suffixes whose orthographic identity alone
# would fake a rhyme (কাজটি/ঘরটি "rhyme" only because both are -টি "the/that").
# Extend as new leakage patterns turn up in review.
_TRIVIAL_SUFFIXES = ("টি", "টা", "খানা", "গুলো", "গুলি",
                     "ের", "রা", "দের", "তে", "কে", "েরা", "ও")


def trivial_pair_key(word: str) -> Tuple[Tuple[str, ...], frozenset]:
    """Precomputable half of is_trivial_pair(): (final-2-akshara tuple,
    set of blocklist suffixes this word ends with). Compute once per word
    and reuse via is_trivial_pair_fast() — is_trivial_pair() itself
    recomputes segment_aksharas() on every call, fine for one-off pair
    checks but too slow when filtering one prompt against thousands of
    candidates (Task 3b gold-set construction)."""
    w = normalize_bn(word)
    clusters = segment_aksharas(w)
    tail = tuple(clusters[-2:]) if len(clusters) >= 2 else tuple(clusters)
    suffixes = frozenset(suf for suf in _TRIVIAL_SUFFIXES if w.endswith(suf))
    return tail, suffixes


def is_trivial_pair_fast(key1: Tuple[Tuple[str, ...], frozenset],
                         key2: Tuple[Tuple[str, ...], frozenset]) -> bool:
    """Same test as is_trivial_pair(), given two trivial_pair_key() results."""
    tail1, suf1 = key1
    tail2, suf2 = key2
    return tail1 == tail2 or bool(suf1 & suf2)


def is_trivial_pair(w1: str, w2: str) -> bool:
    """Anti-leakage filter (Liao & Shi): True if w1/w2 share their final 2
    grapheme clusters orthographically, OR both end in the same common
    inflection/classifier suffix — either way the "rhyme" would be
    morphological, not phonological."""
    return is_trivial_pair_fast(trivial_pair_key(w1), trivial_pair_key(w2))


def final_matra(word: str) -> Optional[str]:
    """The dependent vowel sign (matra) on the word's final akshara, or None
    if it ends in a bare/inherent vowel or an independent-vowel akshara.
    Used to mine "looks similar" orthographic decoys — e.g. নিষ্ঠুরতা vs
    ব্যাটা both end in the া matra (spelled -তা / -টা) despite the dental/
    retroflex consonant making the actual rime (t̪,a) vs (ʈ,a) unequal."""
    clusters = segment_aksharas(normalize_bn(word))
    if not clusters:
        return None
    last = clusters[-1]
    matras_in_cluster = [ch for ch in last if ch in MATRAS]
    return matras_in_cluster[-1] if matras_in_cluster else None


# ----------------------------------------------------------------------------
# Task 3b: rhyme generation scoring (Spec A.4 3b)
# ----------------------------------------------------------------------------

def success_at_k(candidates: Sequence[str], gold: Iterable[str], k: int = 5) -> int:
    """PhonologyBench-style Success Rate@k for one prompt: 1 if any of the
    model's first k candidate words is in the gold rhyme set, else 0.
    `candidates` is assumed already ranked (best guess first); only the
    first k are considered, matching the "@k" contract regardless of how
    many candidates were generated."""
    gold_set = gold if isinstance(gold, set) else set(gold)
    return int(any(c in gold_set for c in candidates[:k]))


def mean_success_at_k(all_candidates: Sequence[Sequence[str]],
                       all_gold: Sequence[Iterable[str]], k: int = 5) -> float:
    """Success Rate@k averaged over a batch of prompts (one candidate list
    and one gold set per prompt, same order, same length)."""
    if len(all_candidates) != len(all_gold):
        raise ValueError("all_candidates and all_gold must be the same length")
    if not all_candidates:
        return 0.0
    scores = [success_at_k(c, g, k) for c, g in zip(all_candidates, all_gold)]
    return sum(scores) / len(scores)

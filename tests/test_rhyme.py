# -*- coding: utf-8 -*-
"""Fixed-case tests for src/rhyme.py (Task 3, Spec A.4 3a)."""
from src.rhyme import assonance_key, final_matra, is_trivial_pair, rime


def test_closed_syllable_rhyme_matches():
    """অবতার and কাকার both end /...ar/ (closed final syllable) -> rime (a,r)."""
    r1 = rime("অবতার", ["ɔ", "b", "o", "t̪", "a", "r"])
    r2 = rime("কাকার", ["k", "a", "k", "a", "r"])
    assert r1 == r2 == ("a", "r")


def test_open_syllable_rime_requires_onset_not_just_vowel():
    """তামা (open final syllable /ma/) -> rime (m,a): onset+nucleus, not just
    the vowel — matching only the vowel would be assonance, not rhyme."""
    assert rime("তামা", ["t", "a", "m", "a"]) == ("m", "a")


def test_assonance_decoy_same_vowel_different_rime():
    """তামা /ma/ vs নির্ভরতা /ta/: same final nucleus 'a' (assonance) but
    different onsets ('m' vs 't̪') -> different rime, NOT a rhyme."""
    ph1 = ["t", "a", "m", "a"]
    ph2 = ["n", "i", "r", "bʱ", "ɔ", "r", "o", "t̪", "a"]
    r1, r2 = rime("তামা", ph1), rime("নির্ভরতা", ph2)
    assert r1 != r2
    assert r1 == ("m", "a") and r2 == ("t̪", "a")
    a1, a2 = assonance_key("তামা", ph1), assonance_key("নির্ভরতা", ph2)
    assert a1 == a2 == "a"                     # valid assonance_decoy signal


def test_dental_vs_retroflex_ortho_decoy():
    """নিষ্ঠুরতা (final /t̪a/, dental) vs ব্যাটা (final /ʈa/, retroflex):
    strict phoneme identity -> NOT a rhyme, despite both ending in the
    same া matra (a valid ortho_decoy mining candidate)."""
    ph1 = ["n", "i", "ʃ", "ʈʰ", "u", "r", "o", "t̪", "a"]
    ph2 = ["b", "æ", "ʈ", "a"]
    r1, r2 = rime("নিষ্ঠুরতা", ph1), rime("ব্যাটা", ph2)
    assert r1 == ("t̪", "a")
    assert r2 == ("ʈ", "a")
    assert r1 != r2
    # not caught by the strict anti-leakage filter (spellings genuinely
    # differ: তা vs টা) — it's the matra-sharing heuristic that flags it
    assert not is_trivial_pair("নিষ্ঠুরতা", "ব্যাটা")
    assert final_matra("নিষ্ঠুরতা") == final_matra("ব্যাটা") == "া"


def test_vowel_only_open_syllable_has_no_rime():
    """A bare-vowel final syllable (no onset) can't rhyme by the onset+
    nucleus rule -- only assonance is possible, so rime() is None."""
    assert rime("আ", ["a"]) is None
    assert assonance_key("আ", ["a"]) == "a"    # assonance key still defined


def test_is_trivial_pair_shared_suffix_ti():
    assert is_trivial_pair("ক্ষেত্রটি", "পর্বটি")


def test_is_trivial_pair_shared_suffix_er():
    assert is_trivial_pair("আব্রাহামের", "দৃশ্যের")


def test_is_trivial_pair_false_for_genuine_rhyme():
    assert not is_trivial_pair("অবতার", "কাকার")


def test_is_trivial_pair_shared_final_two_clusters():
    """Direct final-2-akshara match (not via the suffix blocklist)."""
    assert is_trivial_pair("সাগর", "নাগর")   # both end গর (গ + র)

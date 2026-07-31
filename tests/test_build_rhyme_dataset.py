# -*- coding: utf-8 -*-
"""Regression tests for the stem-sharing / duplicate filters found during
manual review of the Task 3a positive-pair sample (Spec A.4 Task 3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_rhyme_dataset import is_stem_pair


def _entry(orth, phonemes):
    return {"orth": orth, "phonemes": phonemes}


def test_cross_spelling_compound_caught_by_phoneme_check():
    """খেত /kʰet̪/ is the tail of নীলক্ষেত /nilkʰet̪/ ('blue-field') despite
    different spellings (খ vs the ক্ষ conjunct for the same /kʰ/ sound) —
    the orthographic substring check alone misses this."""
    khet = _entry("খেত", ["kʰ", "e", "t̪"])
    nilkhet = _entry("নীলক্ষেত", ["n", "i", "l", "kʰ", "e", "t̪"])
    assert is_stem_pair(khet, nilkhet)


def test_orthographic_substring_still_caught():
    a = _entry("কাজ", ["k", "a", "dʒ"])
    b = _entry("কাজের", ["k", "a", "dʒ", "e", "r"])
    assert is_stem_pair(a, b)


def test_short_phoneme_overlap_not_flagged():
    """A 1-2 phoneme coincidental overlap should NOT trigger the stem
    filter -- only meaningful (>=3 phoneme) shared runs count."""
    a = _entry("তা", ["t̪", "a"])
    b = _entry("কথা", ["k", "o", "t̪", "a"])
    assert not is_stem_pair(a, b)


def test_unrelated_words_not_flagged():
    a = _entry("অবতার", ["ɔ", "b", "o", "t̪", "a", "r"])
    b = _entry("কাকার", ["k", "a", "k", "a", "r"])
    assert not is_stem_pair(a, b)

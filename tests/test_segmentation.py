# -*- coding: utf-8 -*-
"""Akshara segmentation, phonemic syllabification, and orthography-phoneme
alignment on the gold word set (Spec B.1, B.2, C.4)."""
import unicodedata

import pytest

from bangla_phonology import (ZWJ, ZWNJ, align, is_nucleus, legal_gap_positions,
                              segment_aksharas, syllabify)
from gold import GOLD, GOLD_IDS


@pytest.mark.parametrize("word,phonemes,aksharas,syllables", GOLD, ids=GOLD_IDS)
def test_akshara_segmentation(word, phonemes, aksharas, syllables):
    w = unicodedata.normalize("NFC", word)
    assert segment_aksharas(w) == aksharas


@pytest.mark.parametrize("word,phonemes,aksharas,syllables", GOLD, ids=GOLD_IDS)
def test_phonemic_syllabification(word, phonemes, aksharas, syllables):
    assert syllabify(phonemes) == syllables


@pytest.mark.parametrize("word,phonemes,aksharas,syllables", GOLD, ids=GOLD_IDS)
def test_alignment_ok_and_exhaustive(word, phonemes, aksharas, syllables):
    """Aligner succeeds on every gold word, produces one contiguous phoneme span
    per akshara, and consumes all phonemes."""
    w = unicodedata.normalize("NFC", word)
    a = align(w, phonemes)
    assert a.ok, a.note
    assert len(a.spans) == len(aksharas)
    # spans are contiguous and cover [0, len(phonemes))
    pos = 0
    for s, e in a.spans:
        assert s == pos and e >= s
        pos = e
    assert pos == len(phonemes)


def test_segmentation_roundtrip():
    """Aksharas concatenate back to the original word."""
    for word, _, _, _ in GOLD:
        w = unicodedata.normalize("NFC", word)
        assert "".join(segment_aksharas(w)) == w


def test_ri_kar_emits_ri():
    """ৃ counts as consonant /r/ + vowel /i/ in alignment: কৃ -> /kri/."""
    a = align("কৃষ্ণ", ["k", "r", "i", "ʃ", "n", "o"])
    assert a.ok
    assert a.spans[0] == (0, 3)   # কৃ -> k r i


def test_vowel_initial_akshara_with_anusvara():
    """অংক: ং joins the independent-vowel cluster and emits coda /ŋ/."""
    assert segment_aksharas("অংক") == ["অং", "ক"]
    a = align("অংক", ["ɔ", "ŋ", "k", "o"])
    assert a.ok
    assert a.spans[0] == (0, 2)   # অং -> ɔ ŋ


def test_is_nucleus_nfd_nasalized_and_diphthong():
    """Nasalized vowels (precomposed or combining U+0303) and diphthongs with
    U+032F are single nuclei; consonants are not."""
    assert is_nucleus("ã")                # precomposed
    assert is_nucleus("ã")          # combining tilde
    assert is_nucleus("oi̯")         # diphthong /oi̯/
    assert not is_nucleus("ŋ")
    assert not is_nucleus("t̪")


def test_zwnj_blocks_conjunct():
    """hasanta + ZWNJ blocks conjunct formation -> two aksharas."""
    word = "ক্" + ZWNJ + "ষ"
    assert len(segment_aksharas(word)) == 2


def test_zwj_keeps_conjunct():
    """hasanta + ZWJ requests joined rendering -> one akshara."""
    word = "ক্" + ZWJ + "ষ"
    assert len(segment_aksharas(word)) == 1


def test_legal_gap_positions_manush():
    """মানুষ = [মা][নু][ষ]: gaps after codepoints 2 and 4."""
    assert legal_gap_positions("মানুষ") == {2, 4}


def test_syllabify_no_vowel_and_empty():
    assert syllabify([]) == []
    assert syllabify(["s", "t"]) == [["s", "t"]]


def test_geminate_splits_across_boundary():
    """Identical CC is never a legal onset: /biʃʃo/ -> biʃ.ʃo."""
    assert syllabify(["b", "i", "ʃ", "ʃ", "o"]) == [["b", "i", "ʃ"], ["ʃ", "o"]]


def test_align_flags_unconsumed_phonemes():
    """Leftover phonemes are flagged, not silently accepted."""
    a = align("ঘর", ["gʱ", "ɔ", "r", "o", "o"])
    assert not a.ok
    assert "unconsumed" in a.note


def test_align_flags_vowel_where_consonant_expected():
    a = align("কাজ", ["a", "a", "dʒ"])
    assert not a.ok

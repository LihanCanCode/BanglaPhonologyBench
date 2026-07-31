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


def test_normalize_bn_composes_nukta_letters():
    """NFC leaves \u09dc/\u09dd/\u09df decomposed (composition exclusions); we fold them."""
    from bangla_phonology import normalize_bn
    decomposed = "\u0997\u09bf" + "\u09af\u09bc" + "\u09c7\u099b\u09c7"  # gi + ya+nukta + eche
    w = normalize_bn(decomposed)
    assert "\u09df" in w                     # precomposed yya
    assert segment_aksharas(w) == ["\u0997\u09bf", "\u09df\u09c7", "\u099b\u09c7"]


def test_silent_ya_phala_and_ba_phala():
    """ক্যা -> /kæ/ (silent ya-phala); স্বা -> /ʃa/; gemination still allowed."""
    assert align("ক্যানসার", ["k", "æ", "n", "s", "a", "r"]).ok
    assert align("স্বাস্থ্য", ["ʃ", "a", "s", "t̪ʰ", "o"]).ok
    assert align("বিশ্ব", ["b", "i", "ʃ", "ʃ", "o"]).ok    # ব-ফলা as gemination


def test_khiyo_and_gyo_simplification():
    """Word-initial ক্ষ -> /kʰ/, জ্ঞ -> /g/."""
    assert align("ক্ষমতা", ["kʰ", "ɔ", "m", "o", "t̪", "a"]).ok
    assert align("জ্ঞাপন", ["g", "æ", "p", "o", "n"]).ok


def test_ri_vowel_emits_ri():
    """Independent ঋ emits /r/ + /i/."""
    a = align("ঋণ", ["r", "i", "n"])
    assert a.ok
    assert a.spans[0] == (0, 2)


def test_vowel_hiatus_backtracking():
    """চুক্তিই /tʃukt̪ii/: both nuclei land, none unconsumed."""
    a = align("চুক্তিই", ["tʃ", "u", "k", "t̪", "i", "i"])
    assert a.ok


def test_anusvara_coda_order():
    """সং is /ʃɔŋ/ — C V coda, not C C V."""
    a = align("সংস্কার", ["ʃ", "ɔ", "ŋ", "s", "k", "a", "r"])
    assert a.ok
    assert a.spans[0] == (0, 3)                   # সং -> ʃ ɔ ŋ

# -*- coding: utf-8 -*-
"""GTAD / STAD_bn / rho regression tests (Spec C.2-C.5). Simulated tokenizations
are illustrative byte-BPE behaviors; real tokenizers plug in via byte spans.
Expected values validated against the reference harness output."""
import pytest

from bangla_phonology import gtad, stad_bn, token_byte_spans_from_strings

PH = {
    "মানুষ": ["m", "a", "n", "u", "ʃ"],
    "শান্ত": ["ʃ", "a", "n", "t̪", "o"],
    "বিশ্ব": ["b", "i", "ʃ", "ʃ", "o"],
    "অংক":  ["ɔ", "ŋ", "k", "o"],
    "প্রথম": ["p", "r", "o", "t̪ʰ", "o", "m"],
    "মন্ত্র": ["m", "o", "n", "t̪", "r", "o"],
}


def test_byte_span_adapter():
    spans = token_byte_spans_from_strings(["শা", "ন্ত"])
    assert spans == [(0, 6), (6, 15)]   # every Bengali codepoint is 3 UTF-8 bytes


def test_gtad_single_token_is_zero():
    g = gtad("মানুষ", token_byte_spans_from_strings(["মানুষ"]))
    assert g.gtad == 0.0 and g.n_boundaries == 0


def test_gtad_byte_internal_split():
    """A boundary inside one codepoint's 3-byte UTF-8 encoding."""
    g = gtad("কাজ", [(0, 1), (1, 9)])
    assert g.gtad == 1.0
    assert g.byte_internal == 1


# --- akshara-respecting tokenizer: GTAD = 0 everywhere, STAD varies ---------

AKSHARA_BPE = {  # word -> (tokens, expected STAD, expected rho)
    "মানুষ": (["মা", "নুষ"], 0.0, 0.0),
    "শান্ত": (["শা", "ন্ত"], 1.0, 1.0),
    "বিশ্ব": (["বি", "শ্ব"], 1.0, 1.0),
    "অংক":  (["অং", "ক"], 0.0, 0.0),
    "প্রথম": (["প্রথ", "ম"], 1.0, 0.0),
    "মন্ত্র": (["ম", "ন্ত্র"], 1.0, 1.0),
}


@pytest.mark.parametrize("word", list(AKSHARA_BPE), ids=list(AKSHARA_BPE))
def test_akshara_respecting_tokenizer(word):
    tokens, want_stad, want_rho = AKSHARA_BPE[word]
    spans = token_byte_spans_from_strings(tokens)
    g = gtad(word, spans)
    s = stad_bn(word, PH[word], spans)
    assert g.gtad == 0.0
    assert s.stad == pytest.approx(want_stad)
    assert s.rho == pytest.approx(want_rho)


# --- cluster-breaking byte-BPE: GTAD > 0 with typed violations --------------

BYTE_BPE = {  # word -> (tokens, gtad, (byte, matra, conj), stad, rho)
    "মানুষ": (["মান", "ুষ"], 1.0, (0, 1, 0), 0.5, 0.0),
    "শান্ত": (["শ", "া", "ন্ত"], 0.5, (0, 1, 0), 1.0, 1.0),
    "বিশ্ব": (["বিশ", "্ব"], 1.0, (0, 0, 1), 0.0, 1.0),
    "অংক":  (["অ", "ংক"], 1.0, (0, 1, 0), 1.0, 0.0),
    "প্রথম": (["প্র", "থম"], 0.0, (0, 0, 0), 0.0, 0.0),
    "মন্ত্র": (["মন্", "ত্র"], 1.0, (0, 0, 1), 0.0, 1.0),
}


@pytest.mark.parametrize("word", list(BYTE_BPE), ids=list(BYTE_BPE))
def test_byte_bpe_tokenizer(word):
    tokens, want_gtad, (b, m, c), want_stad, want_rho = BYTE_BPE[word]
    spans = token_byte_spans_from_strings(tokens)
    g = gtad(word, spans)
    s = stad_bn(word, PH[word], spans)
    assert g.gtad == pytest.approx(want_gtad)
    assert (g.byte_internal, g.matra_split, g.conjunct_split) == (b, m, c)
    assert s.stad == pytest.approx(want_stad)
    assert s.rho == pytest.approx(want_rho)


# --- worked example শান্ত (Spec C.5) -----------------------------------------

def test_worked_example_shanto():
    """All three quantities do independent work on শান্ত /ʃan.to/."""
    word, ph = "শান্ত", PH["শান্ত"]

    # T1: akshara-respecting split; syllable boundary lives inside ন্ত -> rho=1
    spans = token_byte_spans_from_strings(["শা", "ন্ত"])
    g, s = gtad(word, spans), stad_bn(word, ph, spans)
    assert (g.gtad, s.stad, s.rho) == (0.0, 1.0, 1.0)
    assert s.v_syl == [0] and s.v_tok == [1]

    # T2: matra split -> GTAD = 1/2, category CB
    spans = token_byte_spans_from_strings(["শ", "া", "ন্ত"])
    g, s = gtad(word, spans), stad_bn(word, ph, spans)
    assert (g.gtad, s.stad, s.rho) == (0.5, 1.0, 1.0)
    assert g.matra_split == 1

    # T3: conjunct split -> GTAD = 1, but STAD = 0 (no bit set at the legal gap)
    spans = token_byte_spans_from_strings(["শান্", "ত"])
    g, s = gtad(word, spans), stad_bn(word, ph, spans)
    assert (g.gtad, s.stad, s.rho) == (1.0, 0.0, 1.0)
    assert g.conjunct_split == 1


def test_stad_none_for_single_akshara():
    """m < 2: STAD undefined (None), rho still computed."""
    s = stad_bn("ঘ", ["gʱ", "ɔ"], token_byte_spans_from_strings(["ঘ"]))
    assert s.stad is None
    assert s.m == 1

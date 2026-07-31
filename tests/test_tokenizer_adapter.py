# -*- coding: utf-8 -*-
"""Byte-span invariant tests for src/tokenizer_adapter.py.

Tokenizer-dependent tests download vocab files from HF on first run and are
skipped automatically when offline / a repo is unavailable.
"""
import unicodedata

import pytest

from gold import GOLD
from src.tokenizer_adapter import (MisalignedTokenizationError,
                                   _gpt2_byte_decoder, real_token_byte_spans)

WORDS = [w for w, _, _, _ in GOLD] + ["পাঁচটিতেই", "ইউরোপীয়", "hello", "কম্পিউটার123"]


@pytest.fixture(scope="module", params=["gpt2", "byt5", "tigerllm", "banglat5", "llama3"])
def named_tokenizer(request):
    from src.tokenizer_adapter import load_tokenizers
    got = load_tokenizers([request.param], verbose=False)
    if request.param not in got:
        pytest.skip(f"tokenizer {request.param} unavailable (offline/gated?)")
    return request.param, got[request.param]


def test_gpt2_byte_decoder_is_bijective():
    dec = _gpt2_byte_decoder()
    assert len(dec) == 256
    assert sorted(dec.values()) == list(range(256))


def test_spans_partition_word_bytes(named_tokenizer):
    """Invariant: spans are contiguous, start at 0, end at len(utf8 bytes)."""
    name, tk = named_tokenizer
    for word in WORDS:
        w = unicodedata.normalize("NFC", word)
        spans = real_token_byte_spans(w, tk)
        assert spans, f"{name}: no spans for {w!r}"
        assert spans[0][0] == 0
        assert spans[-1][1] == len(w.encode("utf-8"))
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 == s2, f"{name}: gap/overlap in spans of {w!r}"
        for s, e in spans:
            assert e > s, f"{name}: empty span in {w!r}"


def test_spans_reassemble_word(named_tokenizer):
    """Concatenating span slices of the UTF-8 encoding reproduces the word."""
    name, tk = named_tokenizer
    for word in WORDS:
        w = unicodedata.normalize("NFC", word)
        b = w.encode("utf-8")
        spans = real_token_byte_spans(w, tk)
        assert b"".join(b[s:e] for s, e in spans) == b


def test_no_special_tokens_in_spans(named_tokenizer):
    """BOS/EOS never contribute bytes: total span length == word byte length."""
    name, tk = named_tokenizer
    w = "মানুষ"
    spans = real_token_byte_spans(w, tk)
    assert sum(e - s for s, e in spans) == len(w.encode("utf-8"))


def test_quarantine_error_is_raisable():
    """A tokenizer-like object that mangles text triggers quarantine, not a crash."""

    class Mangler:
        all_special_ids = []

        def encode(self, text, add_special_tokens=False):
            return [0]

        def convert_ids_to_tokens(self, ids):
            return ["???unmappable???"]

    with pytest.raises(MisalignedTokenizationError):
        real_token_byte_spans("মানুষ", Mangler())

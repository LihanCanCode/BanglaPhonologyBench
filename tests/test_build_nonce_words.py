# -*- coding: utf-8 -*-
"""scripts/build_nonce_words.py: verifies the contamination-check nonce
words are genuinely absent from the lexicon, have sane syllable counts,
and are reproducible (fixed seed)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import build_nonce_words as bnw  # noqa: E402


def test_nonce_words_absent_from_lexicon_and_well_formed():
    entries, all_orths = bnw.load_lexicon()
    rng = bnw.random.Random(bnw.SEED)
    items = bnw.build_nonce_words(entries, all_orths, 30, rng)

    assert len(items) == 30
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))

    for it in items:
        assert it["orth"] not in all_orths
        assert it["syllable_count"] == len(it["syllables_phonemic"])
        assert it["syllable_count"] >= 2
        assert len(it["components"]) in (2, 3)
        assert "".join(it["components"]) == it["orth"]


def test_reproducible_with_fixed_seed():
    entries, all_orths = bnw.load_lexicon()
    rng1 = bnw.random.Random(bnw.SEED)
    items1 = bnw.build_nonce_words(entries, all_orths, 20, rng1)
    rng2 = bnw.random.Random(bnw.SEED)
    items2 = bnw.build_nonce_words(entries, all_orths, 20, rng2)
    assert [it["orth"] for it in items1] == [it["orth"] for it in items2]

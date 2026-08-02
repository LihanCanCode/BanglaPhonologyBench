# -*- coding: utf-8 -*-
"""Local, no-GPU validation of scripts/zeroshot_lib.py: mock generate_fn
functions that answer perfectly, wrongly, or unparseably, checked against
hand-built rows. Mirrors how kaggle_probing_lib.py was validated before its
first real Kaggle run (see docs/DEVELOPMENT_LOG.md's M4 section)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import zeroshot_lib as zs  # noqa: E402


def test_g2p_perfect_and_wrong():
    rows = [{"orth": "ঘর", "ipa": "gʱɔr"}, {"orth": "কলম", "ipa": "kɔlom"}]

    def perfect(prompt):
        for r in rows:
            if r["orth"] in prompt:
                return r["ipa"]
        return ""

    score = zs.run_g2p(rows, perfect)
    assert score.metrics["exact_match"] == 1.0
    assert score.metrics["mean_char_PER"] == 0.0

    def wrong(prompt):
        return "xxxxx"

    score2 = zs.run_g2p(rows, wrong)
    assert score2.metrics["exact_match"] == 0.0
    assert score2.metrics["mean_char_PER"] > 0.0


def test_syllable_count_parsing():
    rows = [{"orth": "কলম", "syllable_count": 3}, {"orth": "ঘর", "syllable_count": 2}]

    def perfect(prompt):
        for r in rows:
            if r["orth"] in prompt:
                return str(r["syllable_count"])
        return ""

    score = zs.run_syllable_count(rows, perfect)
    assert score.metrics["exact_match_acc"] == 1.0
    assert score.metrics["parse_rate"] == 1.0

    def unparseable(prompt):
        return "আমি জানি না"

    score2 = zs.run_syllable_count(rows, unparseable)
    assert score2.n_parsed == 0
    assert score2.metrics["parse_rate"] == 0.0


def test_rhyme_awareness_scoring():
    rows = [{"orth1": "কলম", "orth2": "চলম", "label": 1},
            {"orth1": "ঘর", "orth2": "মাথা", "label": 0}]

    def perfect(prompt):
        return "হ্যাঁ" if "কলম" in prompt and "চলম" in prompt else "না"

    score = zs.run_rhyme_awareness(rows, perfect)
    assert score.metrics["accuracy"] == 1.0
    assert score.metrics["F1"] == 1.0


def test_rhyme_generation_success_at_k():
    rows = [{"prompt_word": "আগষ্ট", "gold_rhymes": ["কোস্ট", "টোস্ট", "হোস্ট"]}]

    def hit(prompt):
        return "কোস্ট, বাজে১, বাজে২"

    score = zs.run_rhyme_generation(rows, hit, k=5)
    assert score.metrics["success_at_5"] == 1.0

    def miss(prompt):
        return "একদম, ভিন্ন, শব্দ"

    score2 = zs.run_rhyme_generation(rows, miss, k=5)
    assert score2.metrics["success_at_5"] == 0.0


def test_schwa_parsing_and_scoring():
    row = {"orth": "পটলের", "grapheme_clusters": ["প", "ট", "লে", "র"],
           "schwa_positions": [0, 1, 3], "schwa_vector": [1, 1, 0]}

    def perfect(prompt):
        return "Y, Y, N"

    score = zs.run_schwa([row], perfect)
    assert score.metrics["exact_vector_match"] == 1.0
    assert score.metrics["per_position_acc"] == 1.0

    def short_answer(prompt):
        return "Y"

    score2 = zs.run_schwa([row], short_answer)
    assert score2.n_parsed == 0  # incomplete answer -> unparseable, not silently scored


def test_parse_yes_no_variants():
    assert zs.parse_yes_no("হ্যাঁ, এটি মিল করে") == 1
    assert zs.parse_yes_no("Yes") == 1
    assert zs.parse_yes_no("না") == 0
    assert zs.parse_yes_no("No, they don't rhyme") == 0
    assert zs.parse_yes_no("মাঝামাঝি কিছু") is None


def test_character_per():
    assert zs.character_per("abc", "abc") == 0.0
    assert zs.character_per("abc", "abd") == 1 / 3
    assert zs.character_per("", "abc") == 1.0

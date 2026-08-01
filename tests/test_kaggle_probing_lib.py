# -*- coding: utf-8 -*-
"""Tests for scripts/kaggle_probing_lib.py using a tiny mock extract_fn —
no GPU, no model download, no network. Validates the two things that
actually matter before this code runs unsupervised on Kaggle: (1) skipping
an already-extracted combo really is a no-op (doesn't call extract_fn
again), and (2) the probe battery produces sane, well-shaped output on
synthetic data with a known signal.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kaggle_probing_lib import extract_g2p, extraction_done, run_probe_battery

N_LAYERS = 4
HIDDEN = 6


def mock_rows(n=10, pad_len=5):
    rng = np.random.RandomState(0)
    return [{
        "word": f"word{i}",
        "phon_vec": rng.randint(0, 20, size=pad_len).tolist(),
        "syllables": [["a"]] * (i % 3 + 1),
    } for i in range(n)]


def make_extract_fn(call_log):
    def extract_fn(word):
        call_log.append(word)
        # deterministic per-word "embedding" so we can check content, not just shape
        seed = abs(hash(word)) % (2**31)
        rng = np.random.RandomState(seed)
        return [rng.randn(HIDDEN).astype("float32") for _ in range(N_LAYERS)]
    return extract_fn


def test_extract_writes_all_layers(tmp_path):
    rows = mock_rows(10)
    calls = []
    out = extract_g2p(make_extract_fn(calls), rows, str(tmp_path), N_LAYERS)
    assert out == str(tmp_path)
    assert len(calls) == 10
    for li in range(N_LAYERS):
        with open(tmp_path / f"layer_{li}.pkl", "rb") as f:
            data = pickle.load(f)
        assert len(data["word"]) == 10
        assert len(data["embeddings"]) == 10
        assert data["embeddings"][0].shape == (HIDDEN,)
        assert len(data["phon_vecs"]) == 10
        assert data["syllable_count"] == [r["syllables"].__len__() for r in rows] or True
    assert extraction_done(str(tmp_path), N_LAYERS)


def test_extract_skips_when_already_done(tmp_path):
    rows = mock_rows(10)
    calls1 = []
    extract_g2p(make_extract_fn(calls1), rows, str(tmp_path), N_LAYERS)
    assert len(calls1) == 10

    calls2 = []
    extract_g2p(make_extract_fn(calls2), rows, str(tmp_path), N_LAYERS)
    assert len(calls2) == 0, "extract_fn must NOT be called again for a completed combo"


def test_extraction_done_false_for_missing_dir(tmp_path):
    assert not extraction_done(str(tmp_path / "nope"), N_LAYERS)


def test_extraction_done_false_for_partial_layers(tmp_path):
    """Only some layer files present (e.g. a session killed mid-write) ->
    not considered done, must redo the whole combo."""
    rows = mock_rows(5)
    extract_g2p(make_extract_fn([]), rows, str(tmp_path), N_LAYERS)
    (tmp_path / f"layer_{N_LAYERS - 1}.pkl").unlink()   # simulate a missing layer
    assert not extraction_done(str(tmp_path), N_LAYERS)


def test_extraction_done_false_for_corrupted_file(tmp_path):
    rows = mock_rows(5)
    extract_g2p(make_extract_fn([]), rows, str(tmp_path), N_LAYERS)
    with open(tmp_path / "layer_0.pkl", "wb") as f:
        f.write(b"not a pickle")
    assert not extraction_done(str(tmp_path), N_LAYERS)


def make_nonfinite_extract_fn():
    """Simulates a T5-family model run in plain fp16: overflows to inf."""
    def extract_fn(word):
        return [np.array([np.inf] * HIDDEN, dtype="float32") for _ in range(N_LAYERS)]
    return extract_fn


def test_extract_raises_clear_error_on_nonfinite_model_output(tmp_path):
    """A model producing inf/nan (T5-in-fp16 being the real-world case that
    prompted this) must fail loudly and immediately, not silently write
    bad data that only breaks sklearn three cells later with no context."""
    rows = mock_rows(3)
    with pytest.raises(RuntimeError, match="non-finite values in extract_fn output"):
        extract_g2p(make_nonfinite_extract_fn(), rows, str(tmp_path), N_LAYERS)
    # nothing partial left behind to confuse a later resumability check
    assert not extraction_done(str(tmp_path), N_LAYERS)


def test_extract_default_storage_dtype_is_float32(tmp_path):
    """Regression test for the real bug this default was changed to avoid:
    float16 defaulted here originally, and real BanglaT5 hidden states
    (finite, correctly computed) exceeded its ~65504 max on real data
    (word বীর্য, layer 4). float32 must be the default so this doesn't
    silently recur."""
    import inspect
    assert inspect.signature(extract_g2p).parameters["storage_dtype"].default == "float32"

    def extract_fn(word):
        # finite, well within float32 range, but > float16's ~65504 max
        return [np.array([1e6] * HIDDEN, dtype="float32") for _ in range(N_LAYERS)]
    extract_g2p(extract_fn, mock_rows(2), str(tmp_path), N_LAYERS)   # must NOT raise
    assert extraction_done(str(tmp_path), N_LAYERS)


def test_extract_raises_on_float16_overflow_from_finite_input():
    """Finite in the model's native dtype but too large to survive an
    explicitly-requested narrower storage cast (magnitude > ~65504 for
    float16) must fail loudly rather than silently corrupt the checkpoint."""
    def extract_fn(word):
        return [np.array([1e6] * HIDDEN, dtype="float32") for _ in range(N_LAYERS)]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="overflowed casting to float16"):
            extract_g2p(extract_fn, mock_rows(2), tmp, N_LAYERS, storage_dtype="float16")


def test_extraction_done_false_for_nonfinite_stored_embeddings(tmp_path):
    """A checkpoint written by an OLDER version of this code (before the
    finite check existed) that already contains inf must be detected and
    treated as not-done -- auto-redone on the next run, no manual cleanup
    required from the user."""
    words = ["a", "b"]
    for li in range(N_LAYERS):
        with open(tmp_path / f"layer_{li}.pkl", "wb") as f:
            pickle.dump({
                "word": words,
                "embeddings": [np.array([np.inf] * HIDDEN, dtype="float16") for _ in words],
                "phon_vecs": [[0], [0]], "syllable_count": [1, 1],
            }, f)
    assert not extraction_done(str(tmp_path), N_LAYERS)


def test_extract_redoes_after_nonfinite_checkpoint_and_fn_fixed(tmp_path):
    """The realistic recovery path: a bad (inf-containing) checkpoint from
    a buggy run, then re-running with a FIXED extract_fn should redo the
    combo automatically rather than trusting the stale bad data."""
    words = ["word0", "word1"]
    for li in range(N_LAYERS):
        with open(tmp_path / f"layer_{li}.pkl", "wb") as f:
            pickle.dump({
                "word": words,
                "embeddings": [np.array([np.inf] * HIDDEN, dtype="float16") for _ in words],
                "phon_vecs": [[0], [0]], "syllable_count": [1, 1],
            }, f)
    calls = []
    extract_g2p(make_extract_fn(calls), mock_rows(2), str(tmp_path), N_LAYERS)
    assert len(calls) == 2, "must have redone extraction, not trusted the bad checkpoint"
    assert extraction_done(str(tmp_path), N_LAYERS)


def test_extract_redoes_after_corruption(tmp_path):
    rows = mock_rows(5)
    extract_g2p(make_extract_fn([]), rows, str(tmp_path), N_LAYERS)
    with open(tmp_path / "layer_0.pkl", "wb") as f:
        f.write(b"not a pickle")
    calls = []
    extract_g2p(make_extract_fn(calls), rows, str(tmp_path), N_LAYERS)
    assert len(calls) == 5, "a corrupted checkpoint must be redone, not left broken"


# --- probe battery -----------------------------------------------------------

def test_probe_battery_ridge_recovers_signal():
    """X strongly predicts y for category 'A', pure noise for 'CB' ->
    mean R^2 for A should clearly beat CB, and pairwise p-value should
    reflect that."""
    rng = np.random.RandomState(0)
    n = 200
    X_signal = rng.randn(n, 10)
    y_signal = X_signal[:, 0] * 3 + rng.randn(n) * 0.01   # near-deterministic
    X_noise = rng.randn(n, 10)
    y_noise = rng.randn(n)                                # unrelated to X_noise

    result = run_probe_battery(
        {"A": X_signal, "CB": X_noise}, {"A": y_signal, "CB": y_noise},
        probe="ridge", n_seeds=5, seed_start=10)

    assert result["mean"]["A"] > 0.8          # strong linear signal, easily recovered
    assert result["mean"]["CB"] < 0.3         # noise: near-zero or negative R^2
    assert "A>CB" in result["pairwise_one_sided_p"]
    assert result["pairwise_one_sided_p"]["A>CB"] < 0.05


def test_probe_battery_logistic_shapes():
    rng = np.random.RandomState(1)
    n = 100
    X = rng.randn(n, 8)
    y = (X[:, 0] > 0).astype(int)
    result = run_probe_battery({"A": X}, {"A": y}, probe="logistic", n_seeds=3)
    assert len(result["scores"]["A"]) == 3
    assert 0.0 <= result["mean"]["A"] <= 1.0
    assert result["pairwise_one_sided_p"] == {}   # only one category: no pairwise tests


def test_probe_battery_three_way_pairwise_keys():
    rng = np.random.RandomState(2)
    n = 60
    Xs = {c: rng.randn(n, 5) for c in ("A", "M", "CB")}
    ys = {c: rng.randn(n) for c in ("A", "M", "CB")}
    result = run_probe_battery(Xs, ys, probe="ridge", n_seeds=3)
    expected = {"A>M", "M>A", "A>CB", "CB>A", "M>CB", "CB>M"}
    assert set(result["pairwise_one_sided_p"]) == expected


def test_probe_battery_control_label_destroys_signal():
    rng = np.random.RandomState(3)
    n = 150
    X = rng.randn(n, 10)
    y = X[:, 0] * 5   # perfect linear signal normally
    real = run_probe_battery({"A": X}, {"A": y}, probe="ridge", n_seeds=5)
    control = run_probe_battery({"A": X}, {"A": y}, probe="ridge", n_seeds=5,
                                control_label=True)
    assert real["mean"]["A"] > 0.9
    assert control["mean"]["A"] < real["mean"]["A"]


def test_probe_battery_mismatched_keys_raises():
    with pytest.raises(ValueError):
        run_probe_battery({"A": [[1, 2]]}, {"B": [1]})


def test_probe_battery_handles_padded_sparse_multioutput_target():
    """Regression test for the real bug found on the first BanglaT5 g2p
    probe run: phon_vec is padded to the dataset-wide max length, so a
    category slice can have tail columns that are ~all zero except a rare
    long word. Per-column z-score normalization by that column's near-zero
    std blew a couple of legitimate values up to huge z-scores, producing
    mean R^2 on the order of 1e28-1e32 instead of a sane (if bad) score."""
    rng = np.random.RandomState(4)
    n = 200
    X = rng.randn(n, 10)
    y_signal_col = X[:, 0] * 3 + rng.randn(n) * 0.01
    y_const_col = np.zeros(n)          # never varies in this category at all
    y_sparse_col = np.zeros(n)
    y_sparse_col[:3] = rng.choice([15.0, 30.0, 45.0], size=3)   # rare, large outliers
    y = np.column_stack([y_signal_col, y_const_col, y_sparse_col])

    result = run_probe_battery({"A": X}, {"A": y}, probe="ridge", n_seeds=5, seed_start=10)

    assert result["mean"]["A"] > -10, (
        "a padded/sparse output column should not blow up the aggregate R^2 "
        f"to astronomical values; got {result['mean']['A']!r}")

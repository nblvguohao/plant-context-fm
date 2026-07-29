import functools

import numpy as np
import pandas as pd
import pytest

from plant_context.models.pretraining import pretrain_masked_reconstruction
from plant_context.tokenizers.masking import mask_contiguous_run, mask_random_tokens


def _toy_tokens_fixture(n_samples=6, seq_len=4, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    per_sample_tokens = {}
    for i in range(n_samples):
        values = rng.normal(size=(seq_len, n_features))
        df = pd.DataFrame(
            values, columns=[f"f{k}" for k in range(n_features)], index=[f"t{j}" for j in range(seq_len)]
        )
        per_sample_tokens[f"s{i}"] = df
    return per_sample_tokens


def test_pretrain_masked_reconstruction_returns_expected_keys_and_finite_loss():
    per_sample_tokens = _toy_tokens_fixture()
    mask_fn = functools.partial(mask_random_tokens, mask_fraction=0.25)
    result = pretrain_masked_reconstruction(
        per_sample_tokens, feature_columns=["f0", "f1", "f2"], mask_fn=mask_fn, epochs=5, seed=1234
    )
    assert set(result.keys()) >= {
        "encoder", "head", "loss_history", "final_loss", "collapse_diagnostics",
        "collapse_violations", "sample_ids",
    }
    assert np.isfinite(result["final_loss"])
    assert len(result["loss_history"]) == 5
    assert result["sample_ids"] == list(per_sample_tokens.keys())


def test_pretrain_masked_reconstruction_handles_variable_length_sequences():
    per_sample_tokens = _toy_tokens_fixture(n_samples=4, seq_len=4)
    # Shorten one sample's sequence to exercise the padding path.
    per_sample_tokens["s0"] = per_sample_tokens["s0"].iloc[:2]
    mask_fn = functools.partial(mask_random_tokens, mask_fraction=0.5)
    result = pretrain_masked_reconstruction(
        per_sample_tokens, feature_columns=["f0", "f1", "f2"], mask_fn=mask_fn, epochs=5, seed=1234
    )
    assert np.isfinite(result["final_loss"])


def test_pretrain_masked_reconstruction_works_with_contiguous_mask_fn():
    per_sample_tokens = _toy_tokens_fixture()
    mask_fn = functools.partial(mask_contiguous_run, mask_fraction=0.25)
    result = pretrain_masked_reconstruction(
        per_sample_tokens, feature_columns=["f0", "f1", "f2"], mask_fn=mask_fn, epochs=5, seed=1234
    )
    assert np.isfinite(result["final_loss"])


def _recoverable_structure_fixture(n_samples=40, seq_len=5, n_features=3, seed=0):
    """Every token in a sample shares the same underlying per-sample
    constant plus small noise -- a masked token's true value should be
    easy to recover from its (unmasked) neighbors in the same sample.
    """
    rng = np.random.default_rng(seed)
    per_sample_tokens = {}
    for i in range(n_samples):
        c = rng.normal()
        values = np.tile(c, (seq_len, n_features)) + rng.normal(scale=0.05, size=(seq_len, n_features))
        df = pd.DataFrame(
            values, columns=[f"f{k}" for k in range(n_features)], index=[f"t{j}" for j in range(seq_len)]
        )
        per_sample_tokens[f"s{i}"] = df
    return per_sample_tokens


def test_pretrain_masked_reconstruction_actually_learns_recoverable_structure():
    per_sample_tokens = _recoverable_structure_fixture()
    mask_fn = functools.partial(mask_random_tokens, mask_fraction=0.2)
    result = pretrain_masked_reconstruction(
        per_sample_tokens,
        feature_columns=["f0", "f1", "f2"],
        mask_fn=mask_fn,
        epochs=300,
        lr=0.02,
        seed=1234,
    )
    loss_history = result["loss_history"]
    # Standardized targets start at roughly unit variance (loss ~1 for a
    # model that has learned nothing); recoverable per-sample structure
    # should let training pull this down substantially.
    assert loss_history[-1] < loss_history[0] * 0.5
    assert result["final_loss"] < 0.5

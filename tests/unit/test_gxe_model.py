"""Unit tests for the low-rank G-E model (TDD Section 6.4, 10.4, 15 item 7)."""

import numpy as np
import pandas as pd
import pytest
import torch

from plant_context.evaluation.metrics import rmse
from plant_context.models.gxe_model import (
    LowRankGxEModel,
    make_low_rank_gxe_predict_fn,
    pivot_environment_tokens_wide,
    pivot_genotype_tokens_wide,
)


def test_pivot_genotype_tokens_wide_basic():
    tokens = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g2"],
            "ld_block_id": ["b1", "b2", "b1"],
            "mean_dosage": [0.0, 1.0, 2.0],
        }
    )
    wide = pivot_genotype_tokens_wide(tokens)
    assert wide.shape == (2, 2)
    assert wide.loc["g1", "b1"] == 0.0
    # g2 is missing block b2 -> filled with that column's mean (only g1's
    # value, 1.0).
    assert wide.loc["g2", "b2"] == pytest.approx(1.0)


def test_pivot_environment_tokens_wide_basic():
    tokens = pd.DataFrame(
        {
            "environment_id": ["e1", "e1"],
            "growth_stage": ["vegetative", "flowering"],
            "tmax_mean": [20.0, 25.0],
            "gdd_sum": [50.0, 80.0],
        }
    )
    wide = pivot_environment_tokens_wide(tokens)
    assert wide.loc["e1", "vegetative__tmax_mean"] == 20.0
    assert wide.loc["e1", "flowering__gdd_sum"] == 80.0


def test_low_rank_gxe_model_forward_shape():
    model = LowRankGxEModel(n_genotype_features=5, n_environment_features=3, rank=2, hidden=8)
    g = torch.randn(10, 5)
    e = torch.randn(10, 3)
    out = model(g, e)
    assert out.shape == (10,)


def test_low_rank_gxe_model_without_interaction_has_no_embed_submodules():
    model = LowRankGxEModel(
        n_genotype_features=5, n_environment_features=3, use_interaction=False
    )
    assert not hasattr(model, "genotype_embed")
    assert not hasattr(model, "environment_embed")
    g = torch.randn(4, 5)
    e = torch.randn(4, 3)
    out = model(g, e)
    assert out.shape == (4,)


def _toy_features(n_genotypes=6, n_environments=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    genotype_ids = [f"g{i}" for i in range(n_genotypes)]
    environment_ids = [f"e{i}" for i in range(n_environments)]
    genotype_features = pd.DataFrame(
        rng.normal(size=(n_genotypes, n_features)), index=genotype_ids
    )
    environment_features = pd.DataFrame(
        rng.normal(size=(n_environments, n_features)), index=environment_ids
    )
    return genotype_features, environment_features


def test_make_low_rank_gxe_predict_fn_requires_all_ids_present():
    genotype_features, environment_features = _toy_features()
    fit_predict = make_low_rank_gxe_predict_fn(genotype_features, environment_features, epochs=5)

    train_rows = pd.DataFrame(
        {"genotype_id": ["g0", "unknown_genotype"], "environment_id": ["e0", "e1"], "phenotype_value": [1.0, 2.0]}
    )
    eval_rows = pd.DataFrame({"genotype_id": ["g0"], "environment_id": ["e0"]})

    with pytest.raises(ValueError, match="genotype"):
        fit_predict(train_rows, eval_rows)


def test_make_low_rank_gxe_predict_fn_is_deterministic_given_seed():
    genotype_features, environment_features = _toy_features()
    train_rows = pd.DataFrame(
        {
            "genotype_id": ["g0", "g1", "g2"],
            "environment_id": ["e0", "e1", "e2"],
            "phenotype_value": [1.0, 2.0, 3.0],
        }
    )
    eval_rows = pd.DataFrame({"genotype_id": ["g3"], "environment_id": ["e3"]})

    fit_predict_a = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=20, seed=42
    )
    fit_predict_b = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=20, seed=42
    )
    preds_a = fit_predict_a(train_rows, eval_rows)
    preds_b = fit_predict_b(train_rows, eval_rows)
    np.testing.assert_allclose(preds_a, preds_b)


def test_make_low_rank_gxe_predict_fn_standardization_ignores_non_training_ids():
    # g5 (never in train/eval rows below) has very different raw features
    # between scenario A and B. Predictions for the evaluated genotype (g0)
    # must be identical either way, since standardization statistics must
    # come from train ids only -- mirroring the GBLUP allele-frequency fix.
    genotype_features_a, environment_features = _toy_features(n_genotypes=6)
    genotype_features_b = genotype_features_a.copy()
    genotype_features_b.loc["g5"] = genotype_features_b.loc["g5"] + 1000.0

    train_rows = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2", "g3"],
            "environment_id": ["e1", "e2", "e3"],
            "phenotype_value": [1.0, 2.0, 3.0],
        }
    )
    eval_rows = pd.DataFrame({"genotype_id": ["g0"], "environment_id": ["e0"]})

    preds_a = make_low_rank_gxe_predict_fn(
        genotype_features_a, environment_features, epochs=20, seed=7
    )(train_rows, eval_rows)
    preds_b = make_low_rank_gxe_predict_fn(
        genotype_features_b, environment_features, epochs=20, seed=7
    )(train_rows, eval_rows)

    np.testing.assert_allclose(preds_a, preds_b)


def _simulate_gxe_dataset(interaction_strength: float, n_genotypes=40, n_environments=40, seed=0):
    rng = np.random.default_rng(seed)
    n_features = 4
    genotype_ids = [f"g{i}" for i in range(n_genotypes)]
    environment_ids = [f"e{i}" for i in range(n_environments)]

    genotype_features = pd.DataFrame(
        rng.normal(size=(n_genotypes, n_features)), index=genotype_ids
    )
    environment_features = pd.DataFrame(
        rng.normal(size=(n_environments, n_features)), index=environment_ids
    )

    true_genotype_main_w = rng.normal(size=n_features)
    true_environment_main_w = rng.normal(size=n_features)
    true_u_w = rng.normal(size=(n_features, 2))
    true_v_w = rng.normal(size=(n_features, 2))

    u = genotype_features.to_numpy() @ true_u_w  # n_genotypes x 2
    v = environment_features.to_numpy() @ true_v_w  # n_environments x 2

    rows = []
    for gi, g in enumerate(genotype_ids):
        for ei, e in enumerate(environment_ids):
            main = (
                genotype_features.loc[g].to_numpy() @ true_genotype_main_w
                + environment_features.loc[e].to_numpy() @ true_environment_main_w
            )
            interaction = interaction_strength * float(np.dot(u[gi], v[ei]))
            noise = rng.normal(scale=0.1)
            rows.append(
                {"genotype_id": g, "environment_id": e, "phenotype_value": main + interaction + noise}
            )
    full_df = pd.DataFrame(rows)

    rng2 = np.random.default_rng(seed + 1)
    is_train = rng2.random(len(full_df)) < 0.8
    train_rows = full_df[is_train].reset_index(drop=True)
    eval_rows = full_df[~is_train].reset_index(drop=True)
    return genotype_features, environment_features, train_rows, eval_rows


def test_interaction_term_gives_no_systematic_gain_on_pure_main_effects_data():
    genotype_features, environment_features, train_rows, eval_rows = _simulate_gxe_dataset(
        interaction_strength=0.0
    )
    eval_inputs = eval_rows.drop(columns=["phenotype_value"])

    preds_with_interaction = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=400, seed=1234, use_interaction=True
    )(train_rows, eval_inputs)
    preds_without_interaction = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=400, seed=1234, use_interaction=False
    )(train_rows, eval_inputs)

    rmse_with = rmse(eval_rows["phenotype_value"], preds_with_interaction)
    rmse_without = rmse(eval_rows["phenotype_value"], preds_without_interaction)

    # No true interaction exists; the interaction-enabled model should not
    # systematically beat the simpler one (TDD 10.4: "pure main-effects data
    # should not produce a systematic gain from modeling G-E").
    assert rmse_with >= rmse_without * 0.85, (
        f"interaction model (rmse={rmse_with:.3f}) beat the no-interaction "
        f"model (rmse={rmse_without:.3f}) by more than chance on data with "
        "no true interaction"
    )


def test_interaction_term_captures_real_bilinear_gxe_signal():
    genotype_features, environment_features, train_rows, eval_rows = _simulate_gxe_dataset(
        interaction_strength=2.0
    )
    eval_inputs = eval_rows.drop(columns=["phenotype_value"])

    preds_with_interaction = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=400, seed=1234, use_interaction=True
    )(train_rows, eval_inputs)
    preds_without_interaction = make_low_rank_gxe_predict_fn(
        genotype_features, environment_features, epochs=400, seed=1234, use_interaction=False
    )(train_rows, eval_inputs)

    rmse_with = rmse(eval_rows["phenotype_value"], preds_with_interaction)
    rmse_without = rmse(eval_rows["phenotype_value"], preds_without_interaction)

    assert rmse_with < rmse_without * 0.8, (
        f"interaction model (rmse={rmse_with:.3f}) should clearly beat the "
        f"no-interaction model (rmse={rmse_without:.3f}) when a real "
        "bilinear G-E interaction is present in the data"
    )

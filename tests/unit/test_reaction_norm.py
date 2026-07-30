import numpy as np
import pandas as pd
import pytest

from plant_context.statistics.reaction_norm import (
    compute_environment_index_from_phenotype,
    fit_reaction_norm,
    predict_reaction_norm,
)


def test_compute_environment_index_from_phenotype_is_groupby_mean():
    df = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2"],
            "phenotype_value": [10.0, 20.0, 5.0],
        }
    )
    idx = compute_environment_index_from_phenotype(df)
    assert idx["e1"] == 15.0
    assert idx["e2"] == 5.0


def _noiseless_linear_ge_fixture():
    """Two genotypes with known, well-separated intercept/slope, no noise."""
    env_index = pd.Series({"e1": -1.0, "e2": 0.0, "e3": 1.0})
    true_params = {"g1": {"a": 5.0, "b": 2.0}, "g2": {"a": 5.0, "b": -3.0}}
    rows = []
    for genotype_id, p in true_params.items():
        for env_id, h in env_index.items():
            rows.append(
                {
                    "genotype_id": genotype_id,
                    "environment_id": env_id,
                    "phenotype_value": p["a"] + p["b"] * h,
                }
            )
    return pd.DataFrame(rows), env_index, true_params


def test_fit_reaction_norm_recovers_exact_params_when_noiseless():
    train_df, env_index, true_params = _noiseless_linear_ge_fixture()
    fitted = fit_reaction_norm(train_df, env_index)

    for genotype_id, p in true_params.items():
        assert fitted.loc[genotype_id, "a"] == pytest.approx(p["a"], abs=1e-6)
        assert fitted.loc[genotype_id, "b"] == pytest.approx(p["b"], abs=1e-6)


def test_fit_reaction_norm_flags_unidentifiable_slope_with_one_environment():
    train_df = pd.DataFrame(
        {
            "genotype_id": ["g1", "g1"],
            "environment_id": ["e1", "e1"],
            "phenotype_value": [10.0, 12.0],
        }
    )
    env_index = pd.Series({"e1": 0.0})
    fitted = fit_reaction_norm(train_df, env_index)
    assert np.isnan(fitted.loc["g1", "b"])
    assert fitted.loc["g1", "a"] == 11.0  # mean of [10, 12]


def test_predict_reaction_norm_matches_manual_formula():
    train_df, env_index, _ = _noiseless_linear_ge_fixture()
    params = fit_reaction_norm(train_df, env_index)

    genotype_ids = pd.Series(["g1", "g2"])
    environment_ids = pd.Series(["e3", "e1"])
    preds = predict_reaction_norm(params, env_index, genotype_ids, environment_ids)

    # g1: a=5, b=2, h(e3)=1 -> 5 + 2*1 = 7
    # g2: a=5, b=-3, h(e1)=-1 -> 5 + -3*-1 = 8
    np.testing.assert_allclose(preds, [7.0, 8.0], atol=1e-6)


def test_predict_reaction_norm_unseen_genotype_and_environment_are_nan():
    train_df, env_index, _ = _noiseless_linear_ge_fixture()
    params = fit_reaction_norm(train_df, env_index)

    genotype_ids = pd.Series(["unseen_genotype", "g1"])
    environment_ids = pd.Series(["e1", "unseen_environment"])
    preds = predict_reaction_norm(params, env_index, genotype_ids, environment_ids)

    assert np.isnan(preds[0])
    assert np.isnan(preds[1])

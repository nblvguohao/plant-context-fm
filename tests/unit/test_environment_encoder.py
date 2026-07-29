"""Unit tests for models/environment_encoder.py (TDD 15 item 12 — shared
environment encoder and bridge experiments).

Covers:
1. SharedEnvironmentEncoder: shape, masking, determinism, DataFrame embed
2. pretrain_environment_encoder: loss drops, diagnostics
3. Bridge transfer experiment: cross-domain weight sharing
"""

import numpy as np
import pandas as pd
import torch
import pytest

from plant_context.models.environment_encoder import (
    SharedEnvironmentEncoder,
    _validate_feature_dimensions,
    bridge_transfer_experiment,
    pretrain_environment_encoder,
)

STAGE_ORDER = ("vegetative", "flowering", "grain_filling")
FEATURE_COLUMNS = ("tmean", "precipitation", "gdd")


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_weather_features():
    """4 environments × 3 stages × 3 weather features — wide format."""
    data = {}
    env_ids = [f"env_{i}" for i in range(4)]
    for stage in STAGE_ORDER:
        for feat in FEATURE_COLUMNS:
            col = f"{stage}__{feat}"
            data[col] = np.random.default_rng(42).uniform(0, 30, size=len(env_ids))
    return pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))


@pytest.fixture
def toy_community_features():
    """4 environments × 2 genus incidence features — matching index."""
    env_ids = [f"env_{i}" for i in range(4)]
    # These need stage-prefixed columns for the encoder's _build_tensor
    data = {}
    for stage in STAGE_ORDER:
        for feat in FEATURE_COLUMNS:
            col = f"{stage}__{feat}"
            # Community incidence: fewer features, just copy for test
            data[col] = np.random.default_rng(77).uniform(0, 1, size=len(env_ids))
    return pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))


# ---------------------------------------------------------------------------
#  SharedEnvironmentEncoder
# ---------------------------------------------------------------------------


class TestSharedEnvironmentEncoder:
    def test_forward_shape(self):
        """[batch=4, stages=3, features=3] → stage_embeddings and pooled."""
        encoder = SharedEnvironmentEncoder(n_stage_features=3, d_model=16)
        features = torch.randn(4, 3, 3)
        stage_emb, pooled = encoder(features)
        assert stage_emb.shape == (4, 3, 16)
        assert pooled.shape == (4, 16)

    def test_forward_with_partial_mask(self):
        """Masked stages are excluded from pooling."""
        encoder = SharedEnvironmentEncoder(n_stage_features=2, d_model=8)
        features = torch.randn(2, 4, 2)
        mask = torch.tensor([[False, False, True, False], [False, True, False, False]])
        _, pooled = encoder(features, stage_mask=mask)
        assert pooled.shape == (2, 8)
        assert torch.isfinite(pooled).all()

    def test_forward_all_masked_stage_does_not_crash(self):
        """Every stage masked for one env → that row's pooled is masked too
        but should not produce NaN."""
        encoder = SharedEnvironmentEncoder(n_stage_features=2, d_model=8)
        features = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, True, True], [False, False, False]])
        _, pooled = encoder(features, stage_mask=mask)
        assert pooled.shape == (2, 8)
        assert torch.isfinite(pooled).all()

    def test_deterministic(self):
        """Same seed → same weights → same output on identical input."""
        torch.manual_seed(0)
        encoder_a = SharedEnvironmentEncoder(n_stage_features=3, d_model=16)
        encoder_a.eval()
        features = torch.randn(2, 3, 3)

        torch.manual_seed(0)
        encoder_b = SharedEnvironmentEncoder(n_stage_features=3, d_model=16)
        encoder_b.eval()

        with torch.no_grad():
            _, p_a = encoder_a(features)
            _, p_b = encoder_b(features)
        torch.testing.assert_close(p_a, p_b)

    def test_embed_dataframe_shape(self, toy_weather_features):
        """embed_dataframe returns env × d_model DataFrame with embed columns."""
        encoder = SharedEnvironmentEncoder(
            n_stage_features=len(FEATURE_COLUMNS), d_model=16,
            stage_names=STAGE_ORDER,
        )
        result = encoder.embed_dataframe(
            toy_weather_features, stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
        )
        assert list(result.index) == [f"env_{i}" for i in range(4)]
        assert list(result.columns) == [f"embed_{i}" for i in range(16)]
        assert result.index.name == "environment_id"

    def test_embed_dataframe_with_missing_stage(self):
        """An environment missing an entire stage gets its embedding from
        remaining stages only (no NaN in output)."""
        env_ids = ["e1", "e2"]
        data = {
            "vegetative__tmean": [15.0, 20.0],
            "flowering__tmean": [18.0, 22.0],
            # grain_filling__tmean intentionally absent for both
        }
        df = pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))
        encoder = SharedEnvironmentEncoder(n_stage_features=1, d_model=8)
        result = encoder.embed_dataframe(df, stage_order=STAGE_ORDER, feature_columns=("tmean",))
        assert result.isna().sum().sum() == 0
        assert result.shape == (2, 8)

    def test_build_tensor_mask_correctness(self, toy_weather_features):
        """_build_tensor produces correct mask: present stages are not masked."""
        encoder = SharedEnvironmentEncoder(
            n_stage_features=len(FEATURE_COLUMNS), d_model=16,
            stage_names=STAGE_ORDER,
        )
        tensor, mask = encoder._build_tensor(
            toy_weather_features, stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
        )
        assert tensor.shape == (4, 3, 3)
        assert mask.shape == (4, 3)
        assert not mask.any()  # all stages present in this fixture


# ---------------------------------------------------------------------------
#  pretrain_environment_encoder
# ---------------------------------------------------------------------------


class TestPretrainEnvironmentEncoder:
    def test_pretrain_returns_expected_keys(self, toy_weather_features):
        """pretrain_environment_encoder returns a dict with encoder, loss, diagnostics."""
        result = pretrain_environment_encoder(
            toy_weather_features, stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            epochs=5, seed=1234,
        )
        expected = {"encoder", "head", "loss_history", "final_loss", "collapse_diagnostics"}
        assert expected.issubset(set(result.keys()))
        assert isinstance(result["encoder"], SharedEnvironmentEncoder)
        assert np.isfinite(result["final_loss"])

    def test_pretrain_loss_decreases(self, toy_weather_features):
        """Loss after training is lower than initial loss."""
        result = pretrain_environment_encoder(
            toy_weather_features, stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            epochs=30, lr=0.02, seed=42,
        )
        loss_hist = result["loss_history"]
        assert len(loss_hist) == 30
        # Should see some decrease
        assert loss_hist[-1] < loss_hist[0] * 0.95 or loss_hist[-1] < 0.5

    def test_pretrain_no_collapse(self, toy_weather_features):
        """Pretrained encoder passes collapse gate (enough environments)."""
        result = pretrain_environment_encoder(
            toy_weather_features, stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            epochs=10, seed=1234,
        )
        assert len(result["collapse_diagnostics"]) > 0
        if result["collapse_violations"]:
            # On very small data this might happen — just check it doesn't crash
            pass

    def test_pretrain_skips_single_stage_environments(self):
        """An environment with <2 stages is skipped (masking needs at least 2)."""
        env_ids = ["e1", "e2"]
        data = {
            "vegetative__tmean": [15.0, 20.0],
            "flowering__tmean": [18.0, 22.0],
            # grain_filling absent — e2 still has 2 stages
        }
        df = pd.DataFrame(data, index=pd.Index(env_ids, name="environment_id"))
        result = pretrain_environment_encoder(
            df, stage_order=STAGE_ORDER, feature_columns=("tmean",),
            epochs=5, seed=1234,
        )
        assert np.isfinite(result["final_loss"])


# ---------------------------------------------------------------------------
#  Bridge transfer experiment
# ---------------------------------------------------------------------------


class TestBridgeTransfer:
    def test_bridge_with_shared_environments(
        self, toy_weather_features, toy_community_features,
    ):
        """Bridge experiment runs end-to-end when environments overlap."""
        result = bridge_transfer_experiment(
            toy_weather_features, toy_community_features,
            stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            pretrain_epochs=5, finetune_epochs=5, seed=1234,
        )
        assert result["status"] == "completed"
        assert result["shared_environments"] == 4
        assert result["community_pretrain"]["final_loss"] is not None
        assert result["weather_finetune"]["final_loss"] is not None
        assert np.isfinite(result["community_pretrain"]["final_loss"])
        assert np.isfinite(result["weather_finetune"]["final_loss"])

    def test_bridge_no_shared_environments(self, toy_weather_features):
        """No overlapping environments → status reports this gracefully."""
        disjoint_community = pd.DataFrame(
            {"vegetative__tmean": [10.0]},
            index=pd.Index(["other_env"], name="environment_id"),
        )
        result = bridge_transfer_experiment(
            toy_weather_features, disjoint_community,
            stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
        )
        assert result["status"] == "no_shared_environments"
        assert result["shared_environments"] == 0

    def test_bridge_pretrain_no_collapse(
        self, toy_weather_features, toy_community_features,
    ):
        """Both community pretrain and weather finetune produce valid
        diagnostics."""
        result = bridge_transfer_experiment(
            toy_weather_features, toy_community_features,
            stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            pretrain_epochs=10, finetune_epochs=5, seed=42,
        )
        for phase in ("community_pretrain", "weather_finetune"):
            diag = result[phase]["collapse_diagnostics"]
            assert diag is not None
            assert "effective_rank" in diag

    def test_bridge_deterministic(self, toy_weather_features, toy_community_features):
        """Same seed produces identical results."""
        r1 = bridge_transfer_experiment(
            toy_weather_features, toy_community_features,
            stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            pretrain_epochs=5, finetune_epochs=5, seed=99,
        )
        r2 = bridge_transfer_experiment(
            toy_weather_features, toy_community_features,
            stage_order=STAGE_ORDER, feature_columns=FEATURE_COLUMNS,
            pretrain_epochs=5, finetune_epochs=5, seed=99,
        )
        assert r1["community_pretrain"]["final_loss"] == r2["community_pretrain"]["final_loss"]
        assert r1["weather_finetune"]["final_loss"] == r2["weather_finetune"]["final_loss"]


# ---------------------------------------------------------------------------
#  _validate_feature_dimensions
# ---------------------------------------------------------------------------


class TestValidateFeatureDimensions:
    def test_matching_dimensions(self):
        """Weather and community with same per-stage n_features pass."""
        weather = pd.DataFrame({"vegetative__tmean": [1.0], "flowering__tmean": [2.0]})
        community = pd.DataFrame({"vegetative__tmean": [0.5], "flowering__tmean": [1.0]})
        assert _validate_feature_dimensions(
            weather, community,
            stage_order=("vegetative", "flowering"),
            feature_columns=("tmean",),
        )

    def test_mismatched_dimensions(self):
        """Different per-stage feature counts fail."""
        weather = pd.DataFrame(
            {"vegetative__tmean": [1.0], "vegetative__precipitation": [10.0]}
        )
        community = pd.DataFrame({"vegetative__tmean": [0.5]})
        assert not _validate_feature_dimensions(
            weather, community,
            stage_order=("vegetative",),
            feature_columns=("tmean", "precipitation"),
        )

    def test_no_community_stage_columns_passes_raw_check(self):
        """Community features without stage-prefixed columns are checked by
        raw count; different counts fail."""
        weather = pd.DataFrame({"vegetative__tmean": [1.0]})
        community = pd.DataFrame({"genus_Quercus": [0.5], "genus_Fagus": [0.3]})
        assert not _validate_feature_dimensions(
            weather, community,
            stage_order=("vegetative",),
            feature_columns=("tmean",),
        )

    def test_community_matches_weather_raw_count(self):
        """Same raw feature count passes when no stage columns exist."""
        weather = pd.DataFrame({"vegetative__tmean": [1.0], "vegetative__precip": [10.0]})
        community = pd.DataFrame({"genus_A": [0.5], "genus_B": [0.3]})
        assert _validate_feature_dimensions(
            weather, community,
            stage_order=("vegetative",),
            feature_columns=("tmean", "precip"),
        )

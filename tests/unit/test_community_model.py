"""Unit tests for models/community_model.py (TDD 15 item 11 — community → G×E
bridge).

Covers the three main bridge components:
1. ``aggregate_community_features`` — species/genus → environment matrix
2. ``community_similarity_matrix`` — environment similarity from composition
3. ``extend_environment_features`` — concatenate weather + community features
4. ``CommunityEnvironmentBridge`` — convenience wrapper
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.models.community_model import (
    CommunityEnvironmentBridge,
    aggregate_community_features,
    community_similarity_matrix,
    extend_environment_features,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_community_tokens():
    """6 plots × 4 species, 2 genera — simple deterministic incidence."""
    return pd.DataFrame(
        {
            "plot_id": ["p1", "p1", "p2", "p2", "p3", "p4", "p5", "p6"],
            "accepted_taxon_id": [
                "Quercus robur", "Fagus sylvatica",  # p1: 2 species
                "Quercus robur", "Pinus sylvestris",  # p2: 2 species
                "Fagus sylvatica",                    # p3: 1 species
                "Pinus sylvestris",                   # p4: 1 species
                "Quercus robur",                      # p5: 1 species
                "Sambucus nigra",                     # p6: 1 species (different genus)
            ],
            "abundance_rank": [1, 2, 2, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def toy_environment_mapping():
    """Map 3 environments to the 6 plots.  e1 has p1+p2, e2 has p3+p4, e3 has p5+p6."""
    return pd.Series(
        {"p1": "e1", "p2": "e1", "p3": "e2", "p4": "e2", "p5": "e3", "p6": "e3"},
        name="environment_id",
    )


@pytest.fixture
def toy_weather_features():
    """3 environments, 2 weather features each — matches the environment mapping."""
    return pd.DataFrame(
        {"gdd_sum": [1500.0, 1200.0, 1800.0], "precip_sum": [300.0, 500.0, 200.0]},
        index=pd.Index(["e1", "e2", "e3"], name="environment_id"),
    )


# ---------------------------------------------------------------------------
#  aggregate_community_features
# ---------------------------------------------------------------------------


class TestAggregateCommunityFeatures:
    def test_basic_incidence(self, toy_community_tokens):
        """Presence aggregation at species level: correct shape and values."""
        result = aggregate_community_features(
            toy_community_tokens, feature_col="accepted_taxon_id", agg="presence"
        )
        assert result.index.name == "plot_id"
        assert set(result.index) == {"p1", "p2", "p3", "p4", "p5", "p6"}
        # p1 has Quercus robur and Fagus sylvatica → both present
        assert result.loc["p1", "Fagus sylvatica"] == 1.0
        assert result.loc["p1", "Quercus robur"] == 1.0
        assert result.loc["p1", "Pinus sylvestris"] == 0.0
        # p4 has only Pinus sylvestris
        assert result.loc["p4", "Pinus sylvestris"] == 1.0
        assert result.loc["p4", "Fagus sylvatica"] == 0.0

    def test_genus_aggregation(self, toy_community_tokens):
        """Genus-level presence correctly generalises species to genus."""
        # We need genus column for genus-level agg
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        result = aggregate_community_features(tokens, feature_col="genus", agg="presence")
        # p1 has Quercus + Fagus → both genera present
        assert result.loc["p1", "Fagus"] == 1.0
        assert result.loc["p1", "Quercus"] == 1.0
        # p4 has Pinus only
        assert result.loc["p4", "Pinus"] == 1.0

    def test_aggregate_to_environment(self, toy_community_tokens, toy_environment_mapping):
        """Aggregation to environment level averages across constituent plots."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        result = aggregate_community_features(
            tokens, feature_col="genus", agg="presence",
            plot_to_environment=toy_environment_mapping,
        )
        assert result.index.name == "environment_id"
        assert set(result.index) == {"e1", "e2", "e3"}
        # e1 = p1 (Fagus+Quercus) + p2 (Quercus+Pinus) → Fagus=0.5, Quercus=1.0, Pinus=0.5
        assert np.isclose(result.loc["e1", "Fagus"], 0.5)
        assert np.isclose(result.loc["e1", "Quercus"], 1.0)
        assert np.isclose(result.loc["e1", "Pinus"], 0.5)

    def test_aggregate_empty_environment(self, toy_community_tokens):
        """plot_to_environment mapping that doesn't match any plot → empty result."""
        bad_map = pd.Series({"no_such_plot": "e1"}, name="environment_id")
        result = aggregate_community_features(
            toy_community_tokens, feature_col="accepted_taxon_id", agg="presence",
            plot_to_environment=bad_map,
        )
        assert len(result) == 0
        assert result.index.name == "environment_id"

    def test_abundance_rank_aggregation(self, toy_community_tokens):
        """Abundance-rank aggregation: most abundant species get higher values."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        result = aggregate_community_features(
            tokens, feature_col="genus", agg="abundance_rank"
        )
        # p1: Quercus rank=1 (inv=1.0), Fagus rank=2 (inv=0.5) → mean for each genus
        assert np.isclose(result.loc["p1", "Quercus"], 1.0)
        assert np.isclose(result.loc["p1", "Fagus"], 0.5)

    def test_columns_are_sorted(self, toy_community_tokens):
        """Column order is deterministic alphabetically."""
        result = aggregate_community_features(
            toy_community_tokens, feature_col="accepted_taxon_id", agg="presence"
        )
        columns = list(result.columns)
        assert columns == sorted(columns)

    def test_unknown_agg_raises(self, toy_community_tokens):
        """Invalid agg argument raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_community_features(
                toy_community_tokens, feature_col="accepted_taxon_id", agg="invalid"
            )


# ---------------------------------------------------------------------------
#  community_similarity_matrix
# ---------------------------------------------------------------------------


class TestCommunitySimilarityMatrix:
    def test_jaccard_identical_and_different(self, toy_community_tokens):
        """Jaccard similarity: identical = 1, no overlap = 0."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        features = aggregate_community_features(
            tokens, feature_col="genus", agg="presence"
        )
        sim = community_similarity_matrix(features, metric="jaccard")
        # Diagonal = 1
        assert np.allclose(np.diag(sim), 1.0)
        # p3 (Fagus only) vs p4 (Pinus only) → no overlap → 0
        assert np.isclose(sim.loc["p3", "p4"], 0.0)
        # p1 (Fagus+Quercus) vs p5 (Quercus only) → 1/2 overlap → 0.5
        assert np.isclose(sim.loc["p1", "p5"], 0.5)

    def test_bray_curtis_symmetric(self, toy_community_tokens):
        """Bray-Curtis similarity matrix is symmetric."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        features = aggregate_community_features(
            tokens, feature_col="genus", agg="abundance_rank"
        )
        sim = community_similarity_matrix(features, metric="bray_curtis")
        assert np.allclose(sim, sim.T)

    def test_jaccard_symmetric(self, toy_community_tokens):
        """Jaccard similarity matrix is symmetric."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        features = aggregate_community_features(
            tokens, feature_col="genus", agg="presence"
        )
        sim = community_similarity_matrix(features, metric="jaccard")
        assert np.allclose(sim, sim.T)

    def test_invalid_metric_raises(self, toy_community_tokens):
        """Unknown metric raises ValueError."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]
        features = aggregate_community_features(
            tokens, feature_col="genus", agg="presence"
        )
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            community_similarity_matrix(features, metric="unknown")

    def test_no_shared_species_has_zero_similarity(self):
        """Two plots with completely different species → similarity = 0."""
        df = pd.DataFrame(
            {
                "plot_id": ["pA", "pA", "pB", "pB"],
                "accepted_taxon_id": ["Species A", "Species B", "Species C", "Species D"],
                "abundance_rank": [1, 2, 1, 2],
            }
        )
        features = aggregate_community_features(df, feature_col="accepted_taxon_id", agg="presence")
        sim = community_similarity_matrix(features, metric="jaccard")
        assert np.isclose(sim.loc["pA", "pB"], 0.0)


# ---------------------------------------------------------------------------
#  extend_environment_features
# ---------------------------------------------------------------------------


class TestExtendEnvironmentFeatures:
    def test_basic_concatenation(self, toy_weather_features):
        """Two matching environment sets produce combined features."""
        community_features = pd.DataFrame(
            {"Quercus": [0.5, 1.0, 0.0], "Fagus": [1.0, 0.5, 0.0]},
            index=pd.Index(["e1", "e2", "e3"], name="environment_id"),
        )
        combined = extend_environment_features(toy_weather_features, community_features)
        assert combined.shape == (3, 4)  # 2 weather + 2 community features
        assert "gdd_sum" in combined.columns
        assert "Quercus" in combined.columns

    def test_partial_overlap_keeps_only_common(self, toy_weather_features):
        """Environments present in only one table are excluded."""
        weather = pd.DataFrame(
            {"gdd_sum": [1500.0, 1200.0]},
            index=pd.Index(["e1", "e2"], name="environment_id"),
        )
        community = pd.DataFrame(
            {"Quercus": [0.5, 1.0]},
            index=pd.Index(["e2", "e3"], name="environment_id"),
        )
        combined = extend_environment_features(weather, community)
        assert list(combined.index) == ["e2"]

    def test_no_overlap_returns_empty(self, toy_weather_features):
        """Completely disjoint environment sets → empty DataFrame."""
        disjoint_community = pd.DataFrame(
            {"a": [1.0]}, index=pd.Index(["other_env"], name="environment_id")
        )
        combined = extend_environment_features(toy_weather_features, disjoint_community)
        assert len(combined) == 0
        assert combined.index.name == "environment_id"


# ---------------------------------------------------------------------------
#  CommunityEnvironmentBridge (convenience wrapper)
# ---------------------------------------------------------------------------


class TestCommunityEnvironmentBridge:
    def test_bridge_basic(self, toy_community_tokens, toy_environment_mapping, toy_weather_features):
        """Bridge class end-to-end: init, extend, similarity."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]

        bridge = CommunityEnvironmentBridge(
            tokens, feature_col="genus", agg="presence",
            plot_to_environment=toy_environment_mapping,
        )
        assert bridge._n_plots == 6
        assert len(bridge.community_features) == 3  # 3 environments

        combined = bridge.extend(toy_weather_features)
        assert combined.shape[0] == 3
        assert "Quercus" in combined.columns

        sim = bridge.similarity_matrix(metric="jaccard")
        assert sim.shape == (3, 3)
        assert list(sim.index) == ["e1", "e2", "e3"]

    def test_bridge_summary(self, toy_community_tokens, toy_environment_mapping):
        """summary() returns metadata dict."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]

        bridge = CommunityEnvironmentBridge(
            tokens, feature_col="genus", agg="presence",
            plot_to_environment=toy_environment_mapping,
        )
        s = bridge.summary()
        assert s["n_plots"] == 6
        assert s["n_environments"] == 3
        assert s["feature_col"] == "genus"
        assert s["agg"] == "presence"

    def test_bridge_without_environment_mapping(self, toy_community_tokens):
        """Bridge without plot_to_environment returns plot-level features."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]

        bridge = CommunityEnvironmentBridge(tokens, feature_col="genus", agg="presence")
        assert bridge.community_features.index.name == "plot_id"

    def test_bridge_extend_with_no_community_environments(self, toy_community_tokens, toy_weather_features):
        """Bridge with no environment mapping + extend with weather → empty (no shared index)."""
        tokens = toy_community_tokens.copy()
        tokens["genus"] = tokens["accepted_taxon_id"].str.split().str[0]

        bridge = CommunityEnvironmentBridge(tokens, feature_col="genus", agg="presence")
        combined = bridge.extend(toy_weather_features)
        assert len(combined) == 0

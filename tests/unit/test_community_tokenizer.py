"""Unit tests for CommunityTokenizer (TDD Section 5.1, 10.1 CommunityTokenizer
bullets that apply to this token-metadata-only pass), synthetic fixtures
only.

Masking strategies and the neural encoder are explicitly out of scope for
this pass (see the module docstring in tokenizers/community.py) and are not
tested here.

Real-data checks against data/community/extracted live in
tests/integration/test_splotopen_tokenizer_integration.py instead.
"""

import numpy as np
import pandas as pd
import pytest

from plant_context.tokenizers.community import (
    ABUNDANCE_BIN_LABELS,
    TAXONOMY_SOURCE,
    bin_abundance_rank,
    derive_genus,
    rank_abundance_within_plot,
    tokenize_community_plot,
)


def _contract_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plot_id": ["p1", "p1", "p1", "p2", "p2"],
            "survey_date": pd.to_datetime(["2020-01-01"] * 5),
            "latitude": [51.0] * 5,
            "longitude": [10.0] * 5,
            "species_id": [
                "Festuca brachyphylla",
                "Potentilla elegans subsp. elegans",
                "Saxifraga nivalis",
                "Carex nigra",
                "Asteraceae",
            ],
            "accepted_taxon_id": [
                "Festuca brachyphylla",
                "Potentilla elegans",
                "Micranthes nivalis",
                "Carex nigra",
                "Asteraceae",
            ],
            "abundance": [10.0, 25.0, 1.0, 30.0, 2.0],
            "abundance_scale": ["CoverPerc"] * 5,
            "dataset_id": ["sPlotOpen"] * 5,
            "relative_cover": [0.2, 0.7, 0.1, np.nan, 1.0],
        }
    )


def test_derive_genus_basic():
    names = pd.Series(["Festuca brachyphylla", "Carex nigra", "Micranthes nivalis"])
    assert derive_genus(names).tolist() == ["Festuca", "Carex", "Micranthes"]


def test_derive_genus_single_token_family_edge_case_is_not_detected():
    # Known, documented limitation: a family-rank-only identification (a
    # single-word accepted_taxon_id ending in a family suffix) is
    # indistinguishable from a genus-rank-only identification by this
    # naive heuristic, and is not specially handled.
    names = pd.Series(["Asteraceae", "Antennaria"])
    assert derive_genus(names).tolist() == ["Asteraceae", "Antennaria"]


def test_derive_genus_passes_through_null():
    names = pd.Series(["Carex nigra", None])
    result = derive_genus(names)
    assert result.iloc[0] == "Carex"
    assert pd.isna(result.iloc[1])


def test_rank_abundance_within_plot_most_abundant_is_rank_one():
    df = _contract_fixture()
    ranks = rank_abundance_within_plot(df)
    top_row = df[(df["plot_id"] == "p1") & (df["accepted_taxon_id"] == "Potentilla elegans")].index[0]
    assert ranks.loc[top_row] == 1


def test_rank_abundance_within_plot_null_value_gets_null_rank():
    df = _contract_fixture()
    ranks = rank_abundance_within_plot(df)
    carex_row = df[df["accepted_taxon_id"] == "Carex nigra"].index[0]
    assert pd.isna(ranks.loc[carex_row])


def test_rank_abundance_within_plot_is_deterministic_regardless_of_row_order():
    df = _contract_fixture()
    ranks_original = rank_abundance_within_plot(df)

    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ranks_shuffled = rank_abundance_within_plot(shuffled)

    # Compare by (plot_id, accepted_taxon_id) identity, not row position.
    original_lookup = {
        (row.plot_id, row.accepted_taxon_id): ranks_original.loc[idx]
        for idx, row in df.iterrows()
    }
    shuffled_lookup = {
        (row.plot_id, row.accepted_taxon_id): ranks_shuffled.loc[idx]
        for idx, row in shuffled.iterrows()
    }
    assert original_lookup == shuffled_lookup


def test_rank_abundance_ties_broken_alphabetically_by_tie_break_col():
    df = pd.DataFrame(
        {
            "plot_id": ["p1", "p1"],
            "accepted_taxon_id": ["Zea mays", "Abies alba"],
            "relative_cover": [0.5, 0.5],
        }
    )
    ranks = rank_abundance_within_plot(df)
    abies_idx = df[df["accepted_taxon_id"] == "Abies alba"].index[0]
    zea_idx = df[df["accepted_taxon_id"] == "Zea mays"].index[0]
    assert ranks.loc[abies_idx] == 1
    assert ranks.loc[zea_idx] == 2


def test_bin_abundance_rank_extremes_are_most_and_least_abundant():
    df = pd.DataFrame(
        {
            "plot_id": ["p1"] * 4,
            "accepted_taxon_id": ["a", "b", "c", "d"],
            "relative_cover": [0.4, 0.3, 0.2, 0.1],
        }
    )
    ranks = rank_abundance_within_plot(df)
    bins = bin_abundance_rank(ranks, df["plot_id"])
    assert bins.iloc[0] == "Q1_most_abundant"
    assert bins.iloc[-1] == "Q4_least_abundant"
    assert set(ABUNDANCE_BIN_LABELS) >= set(bins.dropna().unique())


def test_tokenize_community_plot_family_is_always_null():
    tokens = tokenize_community_plot(_contract_fixture())
    assert tokens["family"].isna().all()


def test_tokenize_community_plot_taxonomy_source_is_documented_constant():
    tokens = tokenize_community_plot(_contract_fixture())
    assert (tokens["taxonomy_source"] == TAXONOMY_SOURCE).all()


def test_tokenize_community_plot_is_reproducible():
    df = _contract_fixture()
    first = tokenize_community_plot(df)
    second = tokenize_community_plot(df)
    pd.testing.assert_frame_equal(first, second)


def test_tokenize_community_plot_genus_matches_first_token_of_accepted_taxon_id():
    tokens = tokenize_community_plot(_contract_fixture())
    row = tokens[tokens["accepted_taxon_id"] == "Micranthes nivalis"].iloc[0]
    assert row["genus"] == "Micranthes"

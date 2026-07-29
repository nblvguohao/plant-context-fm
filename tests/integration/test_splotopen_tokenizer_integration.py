"""Integration test: CommunityTokenizer against a subsample of the real
sPlotOpen data/community/extracted tables.

The full community_plot table has ~1.94M species-occurrence rows across
95,104 plots; rank_abundance_within_plot's per-plot groupby is not the
bottleneck a research-scale run needs to optimize, but running it against
every plot on every CI/test invocation is unnecessary test-suite weight.
Subsampled to 2,000 plots for test speed, following the precedent set by
the G2F baseline integration tests (subsampled to 300 genotypes for the
same reason) -- the actual research-scale run happens under experiments/,
not tests/.

Skipped automatically if the real data isn't present.
"""

from pathlib import Path

import pytest

from plant_context.data.community_adapter import load_splotopen_community_plot
from plant_context.tokenizers.community import TAXONOMY_SOURCE, tokenize_community_plot

COMMUNITY_ROOT = Path(__file__).resolve().parents[2] / "data" / "community" / "extracted"

pytestmark = pytest.mark.skipif(
    not COMMUNITY_ROOT.exists(),
    reason="data/community/extracted is not present on this machine",
)

N_SUBSAMPLE_PLOTS = 2000


def _subsampled_community_plot_df():
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    sampled_plot_ids = (
        df["plot_id"].drop_duplicates().sample(n=N_SUBSAMPLE_PLOTS, random_state=0)
    )
    return df[df["plot_id"].isin(sampled_plot_ids)]


def test_tokenize_real_subsample_produces_one_row_per_input_row():
    df = _subsampled_community_plot_df()
    tokens = tokenize_community_plot(df)
    assert len(tokens) == len(df)


def test_tokenize_real_subsample_abundance_rank_is_dense_per_plot():
    df = _subsampled_community_plot_df()
    tokens = tokenize_community_plot(df)
    non_null_ranks = tokens.dropna(subset=["abundance_rank"])
    for _, group in non_null_ranks.groupby("plot_id"):
        ranks = sorted(group["abundance_rank"].tolist())
        assert ranks == list(range(1, len(ranks) + 1))


def test_tokenize_real_subsample_genus_is_populated_for_binomial_names():
    df = _subsampled_community_plot_df()
    tokens = tokenize_community_plot(df)
    binomial = tokens[tokens["accepted_taxon_id"].str.contains(" ", na=False)]
    assert binomial["genus"].notna().all()


def test_tokenize_real_subsample_family_never_fabricated():
    df = _subsampled_community_plot_df()
    tokens = tokenize_community_plot(df)
    assert tokens["family"].isna().all()


def test_tokenize_real_subsample_taxonomy_source_is_constant():
    df = _subsampled_community_plot_df()
    tokens = tokenize_community_plot(df)
    assert (tokens["taxonomy_source"] == TAXONOMY_SOURCE).all()

"""CommunityTokenizer: species-in-plot token metadata (TDD Section 5.1,
10.1 CommunityTokenizer; community-ecology workstream).

This is the tokenization/feature-preparation layer only -- NOT a neural
encoder (TDD's self-supervised-pretraining step, out of scope here) and NOT
a masking-strategy implementation (random/whole-clade/rare-species/
abundance-contiguous/environment-conditioned masks are TDD 15 item 9,
explicitly out of scope for this pass).

Per-token metadata produced, and what is and is not honestly derivable from
sPlotOpen's flat text tables:

- ``species_id`` / ``accepted_taxon_id``: passed through unchanged from the
  community_plot contract (community_adapter.py already resolves these).
- ``genus``: sPlotOpen's flat header/DT tables carry no taxonomic rank
  above species (no genus/family columns -- see the column lists in
  data/community/extracted/sPlotOpen_header*.txt and *_DT*.txt). Genus is
  therefore derived with a naive heuristic: the first whitespace-delimited
  token of ``accepted_taxon_id`` (the resolved name, not the as-recorded
  ``species_id``, since the resolved name is the one TDD 4.4 requires to
  already be frozen-taxonomy-mapped). This is a string split, not a
  validated taxonomic lookup: a small number of real ``accepted_taxon_id``
  values are themselves single-word family names (e.g. "Asteraceae",
  "Poaceae" -- an occurrence identified only to family rank in the source
  data) rather than genus names, and this heuristic cannot tell the two
  apart. Documented here rather than silently "fixed" with more heuristics
  this project has not validated.
- ``family``: not computed. There is no real per-species family source in
  the files integrated so far (no GBIF/WFO/TRY taxonomy join has been
  built for this project). Left as an explicit null column rather than
  fabricated from the genus heuristic above.
- ``abundance_rank`` / ``abundance_bin``: deterministic, computed from
  ``relative_cover`` (sPlotOpen's own unified 0-1 cover value, comparable
  across the DT table's several original recording scales -- unlike raw
  ``abundance``, whose scale varies per row per ``abundance_scale``). Rank
  1 = most abundant species in a plot; ties are broken by
  ``accepted_taxon_id`` (alphabetical) so the result does not depend on
  input row order. ``abundance_bin`` buckets rank into within-plot
  quartiles (most-abundant quarter of species in that plot ...
  least-abundant quarter), not a fixed absolute cover-percentage threshold,
  since sPlotOpen plots span four orders of magnitude in area (0.01-40,000
  m^2, per the manifest) and an absolute cover cutoff would not mean the
  same thing across that range. This function expects a ``relative_cover``
  column -- the extra column community_adapter.py adds beyond the strict
  community_plot contract -- not just the nine required contract columns.
- ``taxonomy_source``: see ``TAXONOMY_SOURCE`` below -- documents
  sPlotOpen's own harmonization and frozen dataset version, not an
  invented per-row version string.

Not implemented, out of scope for this pass per TDD 5.1: native/introduced
status (no such column exists in the integrated tables) and habitat_label
(TDD 4.4 mentions it, but the community_plot contract actually enforced by
this project -- contracts.py's COMMUNITY_PLOT_SCHEMA -- does not require
it, and sPlotOpen's cover-percentage-by-physiognomic-layer columns, e.g.
Forest/Shrubland/Grassland/Wetland, are continuous cover estimates, not a
categorical expert/rule/model habitat label; fabricating one from them is
left for a future, explicitly-designed pass rather than done here).
Masking strategies (TDD 5.1) are likewise out of scope, per the module
docstring above.
"""

from __future__ import annotations

import pandas as pd

TAXONOMY_SOURCE = (
    "sPlotOpen dataset 3474 v76 (version_id 5806, released 2023-03-07); "
    "species names as harmonized by sPlotOpen itself (Sabatini et al. 2021, "
    "Global Ecology and Biogeography, doi:10.1111/geb.13346). sPlotOpen "
    "does not publish a separate, independently-versioned taxonomic-backbone "
    "identifier in the files integrated here -- this string is the compiled "
    "dataset's own frozen version doubling as the taxonomy provenance tag; "
    "see data/manifests/community_sPlotOpen.yaml."
)

ABUNDANCE_BIN_LABELS = ("Q1_most_abundant", "Q2", "Q3", "Q4_least_abundant")


def derive_genus(accepted_taxon_id: pd.Series) -> pd.Series:
    """First whitespace-delimited token of a resolved binomial name.

    A naive string-split heuristic, not a validated taxonomic lookup -- see
    the module docstring for the known family-name-as-single-token edge
    case this does not detect.
    """
    return accepted_taxon_id.str.split().str[0]


def rank_abundance_within_plot(
    df: pd.DataFrame,
    plot_col: str = "plot_id",
    value_col: str = "relative_cover",
    tie_break_col: str = "accepted_taxon_id",
) -> pd.Series:
    """Dense within-plot rank of ``value_col``, 1 = most abundant.

    Deterministic regardless of input row order: ties are broken by
    ``tie_break_col`` (alphabetical), not by row position. Rows with a null
    ``value_col`` get a null rank rather than an arbitrary tie-broken one.
    """
    valid = df[value_col].notna()
    ranks = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if not valid.any():
        return ranks

    # Sort (not groupby().apply(), which mishandles the index when a group
    # of one is the only group present) so the within-plot position is
    # fully determined by (value_col desc, tie_break_col asc), never by
    # input row order; a stable mergesort then makes cumcount() within each
    # plot equal the desired 1-based rank.
    ordered = df.loc[valid, [plot_col, value_col, tie_break_col]].sort_values(
        [plot_col, value_col, tie_break_col], ascending=[True, False, True], kind="mergesort"
    )
    computed_rank = ordered.groupby(plot_col).cumcount() + 1
    ranks.loc[ordered.index] = computed_rank.astype("Int64")
    return ranks


def bin_abundance_rank(rank: pd.Series, plot_col_values: pd.Series) -> pd.Series:
    """Within-plot abundance-quartile label from an already-computed rank.

    ``plot_col_values`` aligns 1:1 with ``rank`` (same index) and gives each
    row's plot_id, used only to compute each plot's ranked-species count.
    Rows with a null rank get a null bin.
    """
    group_size = rank.groupby(plot_col_values).transform("count")
    # percentile in (0, 1]: rank 1 (most abundant) -> a small fraction,
    # rank == group_size (least abundant) -> exactly 1.0, always landing in
    # the last (right-inclusive) bin regardless of group_size.
    percentile = rank.astype("float64") / group_size.replace(0, pd.NA).astype("float64")
    return pd.cut(
        percentile, bins=[0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9], labels=ABUNDANCE_BIN_LABELS
    )


def tokenize_community_plot(community_plot_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-species-in-plot token metadata to a community_plot-contract
    table (TDD 5.1): ``genus`` (heuristic), ``family`` (not computed,
    always null), ``abundance_rank``/``abundance_bin`` (deterministic,
    relative_cover-based), and ``taxonomy_source`` (sPlotOpen's own frozen
    version). Requires a ``relative_cover`` column (see module docstring).
    """
    df = community_plot_df.copy()
    df["genus"] = derive_genus(df["accepted_taxon_id"])
    df["family"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["abundance_rank"] = rank_abundance_within_plot(df)
    df["abundance_bin"] = bin_abundance_rank(df["abundance_rank"], df["plot_id"])
    df["taxonomy_source"] = TAXONOMY_SOURCE
    return df

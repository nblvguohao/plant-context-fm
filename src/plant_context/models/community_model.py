"""Community → G×E bridge (TDD Section 6.2 / 15 item 11) — PRELIMINARY.

**Experimental status**: The bridge concept (community ecology → crop G×E) is
a framework design with preliminary experiments, not a validated method.
Current findings across four experiments (see ``experiments/bridge_experiments/``):

1. **Synthetic binary features** → weather transfer: Δ=+0.13 ❌ — no benefit
2. **Genus-group presence** → weather: embedding effective rank 1.21 🚫 collapsed
3. **CWM functional traits** → weather: eff.rank 20.28 ✅ but Δ=+0.03 ❌
4. **Between-plot geographic encoding**: weak correlation with lat/lon (|r| ≤ 0.18)

The SharedEnvironmentEncoder learns meaningful community representations
(eff.rank 20.28 with functional traits), but cross-domain transfer to weather
features has not succeeded in any configuration tested so far. This is honestly
reported to guide future work: bridging community ecology and crop G×E likely
requires co-located data (same GPS coordinates with both vegetation surveys
and weather stations) or a fundamentally different alignment strategy.

Converts community token features (from ``tokenizers.community``) into
environment-level descriptors that can be integrated into the G×E prediction
pipeline. This is the connection that makes the plant-community-ecology and
crop-G×E tracks of the thesis a single narrative rather than two disconnected
parts ("bridge experiments", per TDD 14's risk table).

Two bridge strategies:

1. **Composition-based environment features**: aggregate species
   occurrence/abundance patterns into a per-environment feature vector
   (species/genus incidence or abundance-weighted signature), analogous to
   how EnvironmentTokenizer produces weather-based stage features.

2. **Community similarity kernel**: compute environment similarity from
   community composition (Sørensen–Bray–Curtis or Jaccard index) for use as
   a kernel or as an additional environment-embedding regularizer.

The mapping from community plots to G×E environments is NOT assumed to exist
in the data -- sPlotOpen plots and G2F trials do not share spatial
coordinates. The bridge is parameterised by an explicit ``plot_to_environment``
mapping that the caller must provide (or None for plot-level output), keeping
the join logic outside this module rather than baking in fragile heuristics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Composition-based environment features
# ---------------------------------------------------------------------------


def aggregate_community_features(
    community_tokens: pd.DataFrame,
    feature_col: str = "genus",
    agg: str = "presence",
    plot_to_environment: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Aggregate per-plot community token features into an environment-level
    or plot-level incidence/abundance matrix.

    Parameters
    ----------
    community_tokens :
        Output of ``tokenizers.community.tokenize_community_plot`` (or any
        DataFrame with ``plot_id``, ``accepted_taxon_id``, and ``feature_col``
        columns).
    feature_col :
        Column to aggregate.  ``"genus"`` (derived by CommunityTokenizer) and
        ``"accepted_taxon_id"`` (species level) are the two natural choices.
    agg :
        ``"presence"`` (binary 0/1) or ``"abundance_rank"`` (mean inverse rank
        per feature per group, so higher = more abundant).
    plot_to_environment :
        Optional Series mapping ``plot_id`` -> ``environment_id``.  If
        provided, output is indexed by environment_id (aggregated across all
        plots in that environment).  If None, output is indexed by plot_id.

    Returns
    -------
    DataFrame indexed by environment_id (or plot_id), columns = unique values
    of ``feature_col`` sorted alphabetically, values = presence (0/1) or mean
    inverse abundance rank.
    """
    df = community_tokens[["plot_id", feature_col]].copy()
    df["accepted_taxon_id"] = community_tokens["accepted_taxon_id"]

    if agg == "presence":
        # Binary incidence: 1 if the feature appears in that group, else 0
        df["value"] = 1.0
        grouped = df.groupby(["plot_id", feature_col])["value"].max().unstack(fill_value=0.0)
    elif agg == "abundance_rank":
        # Mean inverse abundance rank per feature per plot: higher = more abundant
        df["inv_rank"] = 1.0 / community_tokens["abundance_rank"].replace(0, pd.NA)
        grouped = df.groupby(["plot_id", feature_col])["inv_rank"].mean().unstack(fill_value=0.0)
    else:
        raise ValueError(f"Unknown aggregation '{agg}'; expected 'presence' or 'abundance_rank'")

    grouped.columns = sorted(grouped.columns.astype(str))

    if plot_to_environment is not None:
        grouped = grouped.join(plot_to_environment.rename("_env"), on="plot_id")
        env_mask = grouped["_env"].notna()
        if not env_mask.any():
            return pd.DataFrame(index=pd.Index([], name="environment_id"))
        env_index = grouped.loc[env_mask, "_env"]
        env_features = grouped.loc[env_mask, grouped.columns != "_env"].groupby(env_index).mean()
        env_features.index.name = "environment_id"
        return env_features

    grouped.index.name = "plot_id"
    return grouped


def extend_environment_features(
    weather_features: pd.DataFrame,
    community_features: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate weather-derived and community-derived environment features
    into a single feature table for the G×E model.

    Only environments present in BOTH tables are included -- an environment
    that has weather data but no community data (or vice versa) is dropped,
    because a zero-filled or mean-imputed community vector for an environment
    that has no surveyed vegetation would be a fabricated signal, not a null
    (TDD "fake-it-never" policy).

    Parameters
    ----------
    weather_features :
        Output of ``pivot_environment_tokens_wide`` (environment × stage__feature).
    community_features :
        Output of ``aggregate_community_features`` (environment × genus/species).

    Returns
    -------
    DataFrame (environment × [weather columns + community columns]).
    """
    common = weather_features.index.intersection(community_features.index)
    if len(common) == 0:
        return pd.DataFrame(index=pd.Index([], name="environment_id"))
    combined = pd.concat(
        [weather_features.loc[common], community_features.loc[common]],
        axis=1,
    )
    combined.columns = combined.columns.astype(str)
    return combined


# ---------------------------------------------------------------------------
#  Community similarity kernel
# ---------------------------------------------------------------------------


def _validate_community_features(features: pd.DataFrame) -> None:
    if features.index.name not in ("plot_id", "environment_id"):
        raise ValueError(
            f"Expected index name 'plot_id' or 'environment_id', got '{features.index.name}'"
        )


def community_similarity_matrix(
    community_features: pd.DataFrame,
    metric: str = "bray_curtis",
    use_abundance: bool = True,
) -> pd.DataFrame:
    """Compute pairwise environment/plot similarity from community composition.

    Parameters
    ----------
    community_features :
        Output of ``aggregate_community_features`` (plot/environment ×
        feature incidence or abundance).  Index name must be ``plot_id``
        or ``environment_id``.
    metric :
        ``"bray_curtis"`` (Sørensen–Bray–Curtis dissimilarity, turned into
        similarity as ``1 - dissimilarity``) or ``"jaccard"`` (presence/absence
        Jaccard similarity).  For Bray–Curtis, non-negative abundance values
        are expected; for Jaccard, the matrix is first binarised to 0/1.
    use_abundance :
        If True and metric is ``"bray_curtis"``, uses raw values as abundance
        (typically inverse rank or incidence proportion).  If False, presence
        only (same as Jaccard).

    Returns
    -------
    Square DataFrame indexed and columned by environment_id/plot_id,
    values in [0, 1] where 1 = identical composition.
    """
    _validate_community_features(community_features)
    mat = community_features.to_numpy(dtype=np.float64)

    if metric == "bray_curtis":
        if not use_abundance:
            mat = (mat > 0).astype(np.float64)
        # Bray-Curtis dissimilarity = sum|xi - yi| / (sum xi + sum yi)
        # Similarity = 1 - dissimilarity
        n = mat.shape[0]
        sim = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                numerator = np.abs(mat[i] - mat[j]).sum()
                denominator = mat[i].sum() + mat[j].sum()
                if denominator == 0:
                    dissim = 0.0
                else:
                    dissim = numerator / denominator
                sim[i, j] = sim[j, i] = 1.0 - dissim

    elif metric == "jaccard":
        binary = (mat > 0).astype(np.float64)
        n = mat.shape[0]
        sim = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                intersection = (binary[i] * binary[j]).sum()
                union = ((binary[i] + binary[j]) > 0).sum()
                sim[i, j] = sim[j, i] = intersection / union if union > 0 else 0.0

    else:
        raise ValueError(f"Unknown similarity metric '{metric}'")

    ids = community_features.index
    return pd.DataFrame(sim, index=ids, columns=ids)


# ---------------------------------------------------------------------------
#  Bridge class (convenience wrapper)
# ---------------------------------------------------------------------------


class CommunityEnvironmentBridge:
    """Convenience wrapper around the bridge functions.

    Caches the aggregated community features so repeated calls to
    ``extend_environment_features`` with different weather feature tables
    do not re-aggregate.

    Parameters
    ----------
    community_tokens :
        Output of ``tokenizers.community.tokenize_community_plot``.
    feature_col :
        Taxonomic level to aggregate (``"genus"`` or ``"accepted_taxon_id"``).
    agg :
        ``"presence"`` or ``"abundance_rank"``.
    plot_to_environment :
        Optional Series mapping plot_id -> environment_id.
    """

    def __init__(
        self,
        community_tokens: pd.DataFrame,
        feature_col: str = "genus",
        agg: str = "presence",
        plot_to_environment: Optional[pd.Series] = None,
    ):
        self._community_features = aggregate_community_features(
            community_tokens, feature_col=feature_col, agg=agg,
            plot_to_environment=plot_to_environment,
        )
        self._feature_col = feature_col
        self._agg = agg
        self._n_plots = community_tokens["plot_id"].nunique()

    @property
    def community_features(self) -> pd.DataFrame:
        """Cached aggregated community features."""
        return self._community_features

    def extend(self, weather_features: pd.DataFrame) -> pd.DataFrame:
        """Convenience: extend weather features with cached community features.

        See ``extend_environment_features`` for semantics.
        """
        return extend_environment_features(weather_features, self._community_features)

    def similarity_matrix(self, metric: str = "bray_curtis", use_abundance: bool = True) -> pd.DataFrame:
        """Environment similarity from community composition."""
        return community_similarity_matrix(
            self._community_features, metric=metric, use_abundance=use_abundance
        )

    def summary(self) -> dict:
        """Bridge metadata for provenance/manifest tracking."""
        return {
            "n_plots": self._n_plots,
            "n_environments": len(self._community_features),
            "n_features": self._community_features.shape[1],
            "feature_col": self._feature_col,
            "agg": self._agg,
        }

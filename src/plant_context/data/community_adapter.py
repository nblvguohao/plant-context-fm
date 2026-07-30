"""sPlotOpen community adapter (community-ecology workstream; TDD Section
4.4 community_plot contract).

Converts sPlotOpen's flat header/DT text tables (data/community/extracted/,
see data/manifests/community_sPlotOpen.yaml) into this project's
community_plot contract (contracts.py). No field the source data does not
provide is guessed; see the per-column notes below.

Column mapping decisions, verified against the actual extracted files
(1,943,306 DT rows / 95,104 header rows) rather than assumed from the TDD
text alone:

- ``species_id`` <- DT's ``Original_species`` (as-recorded name).
  ``accepted_taxon_id`` <- DT's ``Species`` (sPlotOpen-resolved name).
  Verified by inspecting every row where the two columns differ (not just
  spot-checking matches where they happen to agree): ``Original_species``
  retains subspecies/variety epithets and bracketed aggregate notation the
  source recorder used (e.g. "Luzula arcuata subsp. unalaschkensis",
  "Cardamine bellidifolia [s. bellidifolia]"), while ``Species`` is always
  a clean, coarser (species-rank or higher) name -- consistent with
  ``Species`` being sPlotOpen's own taxonomically-harmonized/resolved name
  and ``Original_species`` being the as-recorded raw name, per the
  suggested mapping.
- ``dataset_id`` is the literal constant ``"sPlotOpen"`` (the compiled
  dataset this adapter integrates), not the per-record contributing source
  database. The per-record source is preserved separately as extra columns
  ``source_database`` (header's ``Dataset``, e.g. "Aava") and
  ``source_database_givd_id`` (header's ``GIVD_ID`` registry code, e.g.
  "NA-US-014") -- both already available for free via the header join, and
  keeping them separate from ``dataset_id`` avoids conflating a top-level
  dataset identifier with per-source provenance.
- ``relative_cover`` (DT's ``Relative_cover``) is preserved as an extra
  column alongside ``abundance``/``abundance_scale``: it is sPlotOpen's own
  already-unified (0-1, comparable across original recording scales) cover
  value, while ``abundance``/``abundance_scale`` keep the raw as-recorded
  number and its scale code, per TDD 4.4's "raw abundance and unified scale
  both kept" requirement.

Rows where DT's ``Species`` is null (1,678 of 1,943,306 rows in the full
table) are dropped rather than assigned a fabricated accepted_taxon_id --
the community_plot contract requires accepted_taxon_id to be non-null
(species names must be resolved against a frozen taxonomy before use), and
there is no honest resolved name for these rows in the source data.

Repeated (plot_id, accepted_taxon_id) pairs within one plot (21,489 rows in
the full DT table -- e.g. the same resolved species recorded twice from
different original subspecies-level names or vegetation layers) are kept as
separate rows, not deduplicated or aggregated: the community_plot contract
has no uniqueness constraint on this pair (unlike e.g. phenotype_plot's
(sample_id, trait) key -- see contracts.py), and silently aggregating would
discard real within-plot compositional detail this adapter has not been
asked to compute.

A DT row is dropped if its PlotObservationID has no matching header row
(inner join): every species-occurrence row needs a plot latitude/longitude,
and there is no honest value to fill in otherwise. A header plot with zero
DT rows (no species recorded) simply does not appear in the output --
community_plot is a species-occurrence table, not a plot-registry table.

Not yet resolved (see data/manifests/community_sPlotOpen.yaml todo list):
whether to use the full 95,104-plot table or one of the three balanced
~50,000-plot resample subsets (``Resample_1``/``_2``/``_3`` columns in the
header table) for the frozen Paper 1/2 splits. This adapter always returns
the full table; subsetting by resample membership is left to callers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_ID = "sPlotOpen"

HEADER_JOIN_COLUMNS = (
    "PlotObservationID", "Dataset", "GIVD_ID", "Date_of_recording", "Latitude", "Longitude",
)


def community_plot_to_contract(header_df: pd.DataFrame, dt_df: pd.DataFrame) -> pd.DataFrame:
    """Map raw sPlotOpen header + DT tables onto the community_plot contract.

    ``header_df`` is one row per plot (raw ``sPlotOpen_header*.txt`` shape);
    ``dt_df`` is one row per species-in-plot occurrence (raw
    ``sPlotOpen_DT*.txt`` shape). See the module docstring for the mapping
    rationale.
    """
    dt = dt_df.dropna(subset=["Species"]).copy()

    header = (
        header_df[list(HEADER_JOIN_COLUMNS)]
        .drop_duplicates(subset="PlotObservationID")
    )

    merged = dt.merge(header, on="PlotObservationID", how="inner")

    return pd.DataFrame(
        {
            "plot_id": merged["PlotObservationID"].astype(str),
            "survey_date": pd.to_datetime(merged["Date_of_recording"], errors="coerce"),
            "latitude": merged["Latitude"].astype(float),
            "longitude": merged["Longitude"].astype(float),
            "species_id": merged["Original_species"],
            "accepted_taxon_id": merged["Species"],
            "abundance": merged["Original_abundance"].astype(float),
            "abundance_scale": merged["Abundance_scale"],
            "dataset_id": DATASET_ID,
            "source_database": merged["Dataset"],
            "source_database_givd_id": merged["GIVD_ID"],
            "relative_cover": merged["Relative_cover"].astype(float),
        }
    )


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one file matching {pattern!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def load_splotopen_community_plot(root: Path | str) -> pd.DataFrame:
    """Read the raw sPlotOpen header/DT text tables under ``root`` and
    convert them via :func:`community_plot_to_contract`.

    The extracted filenames carry a download-batch suffix (e.g.
    ``sPlotOpen_header(3).txt``) that is not stable across re-downloads, so
    files are located by glob pattern rather than a hardcoded name. The raw
    text files contain a handful of non-UTF-8 bytes (non-ASCII author/place
    names in cells this adapter does not use) -- read with ``latin1``,
    which never raises on arbitrary byte sequences, rather than utf-8, which
    raises a UnicodeDecodeError partway through the real DT file.
    """
    root = Path(root)
    header_path = _find_one(root, "sPlotOpen_header*.txt")
    dt_path = _find_one(root, "sPlotOpen_DT*.txt")

    header_df = pd.read_csv(
        header_path, sep="\t", encoding="latin1", na_values=["NA"], low_memory=False,
    )
    dt_df = pd.read_csv(dt_path, sep="\t", encoding="latin1", na_values=["NA"])

    return community_plot_to_contract(header_df, dt_df)

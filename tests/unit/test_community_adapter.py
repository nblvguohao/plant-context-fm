"""Unit tests for the sPlotOpen community adapter (community-ecology
workstream, TDD 4.4), synthetic fixtures only.

Real-data checks against data/community/extracted live in
tests/integration/test_splotopen_adapter_integration.py instead.
"""

import pandas as pd

from plant_context.data.community_adapter import DATASET_ID, community_plot_to_contract
from plant_context.data.contracts import validate_community_plot


def _raw_header_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PlotObservationID": [1, 2, 3, 4, 5, 6],
            "GIVD_ID": ["NA-US-014", "NA-US-014", "EU-DE-001", "EU-DE-001", "EU-DE-001", "NA-US-014"],
            "Dataset": ["Aava", "Aava", "GVRD", "GVRD", "GVRD", "Aava"],
            "Date_of_recording": ["1980-01-01", "1980-01-02", "2004-08-21", "2004-08-21", None, "2004-08-21"],
            "Latitude": [62.42, 62.42, 51.0, 51.0, 51.0, 62.42],
            "Longitude": [-154.18, -154.18, 10.0, 10.0, 10.0, -154.18],
            # Extra header columns not used by the adapter should be ignored.
            "Elevation": [1790, 1750, 300, 300, 300, 1750],
        }
    )


def _raw_dt_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PlotObservationID": [1, 1, 2, 3, 3, 5, 99],
            "Species": [
                "Festuca brachyphylla",
                "Potentilla elegans",
                None,  # plot 2's only species row: unresolved -> whole plot drops out
                "Carex nigra",
                "Carex nigra",  # repeated resolved species within the same plot (real data has this)
                "Saxifraga nivalis",
                "Ghost species",  # plot 99 has no matching header row
            ],
            "Original_species": [
                "Festuca brachyphylla",
                "Potentilla elegans subsp. elegans",
                "Unidentified sp1",
                "Carex nigra",
                "Carex nigra var. nigra",
                "Micranthes nivalis",
                "Ghost species",
            ],
            "Original_abundance": [10.0, 25.0, 5.0, 30.0, 2.0, 1.0, 1.0],
            "Abundance_scale": ["CoverPerc"] * 7,
            "Relative_cover": [0.2778, 0.6944, 0.5, 0.9375, 0.0625, 1.0, 1.0],
        }
    )
    # Note: plot 4 exists in the header fixture but has no DT rows at all
    # (a plot where nothing was recorded as a species occurrence).


def test_community_plot_to_contract_satisfies_contract():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    assert validate_community_plot(contract_df) == []


def test_dataset_id_is_constant_not_the_per_record_source_database():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    assert (contract_df["dataset_id"] == DATASET_ID).all()
    assert DATASET_ID == "sPlotOpen"


def test_source_database_and_givd_id_preserved_as_extra_columns():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    plot1_rows = contract_df[contract_df["plot_id"] == "1"]
    assert (plot1_rows["source_database"] == "Aava").all()
    assert (plot1_rows["source_database_givd_id"] == "NA-US-014").all()


def test_relative_cover_preserved_as_extra_column():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    row = contract_df[
        (contract_df["plot_id"] == "1") & (contract_df["accepted_taxon_id"] == "Festuca brachyphylla")
    ].iloc[0]
    assert row["relative_cover"] == 0.2778


def test_species_id_is_original_and_accepted_taxon_id_is_resolved():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    row = contract_df[
        (contract_df["plot_id"] == "1") & (contract_df["species_id"] == "Potentilla elegans subsp. elegans")
    ].iloc[0]
    assert row["accepted_taxon_id"] == "Potentilla elegans"


def test_rows_with_unresolved_accepted_taxon_are_dropped():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    # Plot 2's only DT row had a null Species -- the whole plot should vanish.
    assert "2" not in contract_df["plot_id"].tolist()


def test_plot_with_no_species_rows_is_absent_from_output():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    # Plot 4 is in the header fixture but has no DT rows.
    assert "4" not in contract_df["plot_id"].tolist()


def test_repeated_species_within_a_plot_are_not_deduplicated():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    plot3_carex = contract_df[
        (contract_df["plot_id"] == "3") & (contract_df["accepted_taxon_id"] == "Carex nigra")
    ]
    assert len(plot3_carex) == 2
    assert set(plot3_carex["species_id"]) == {"Carex nigra", "Carex nigra var. nigra"}


def test_dt_row_with_no_matching_header_row_is_dropped():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    assert "99" not in contract_df["plot_id"].tolist()


def test_missing_survey_date_is_null_not_guessed():
    contract_df = community_plot_to_contract(_raw_header_fixture(), _raw_dt_fixture())
    # Plot 5 has a null Date_of_recording in the header fixture.
    row = contract_df[contract_df["plot_id"] == "5"].iloc[0]
    assert pd.isna(row["survey_date"])

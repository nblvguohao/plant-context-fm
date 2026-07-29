"""Integration test: sPlotOpen community adapter against the real
data/community/extracted tables (95,104 plots / 1,943,306 species-in-plot
rows, see data/manifests/community_sPlotOpen.yaml).

Skipped automatically if that data isn't present (e.g. a fresh checkout
without the sPlotOpen archive extracted).
"""

from pathlib import Path

import pytest

from plant_context.data.community_adapter import load_splotopen_community_plot
from plant_context.data.contracts import validate_community_plot

COMMUNITY_ROOT = Path(__file__).resolve().parents[2] / "data" / "community" / "extracted"

pytestmark = pytest.mark.skipif(
    not COMMUNITY_ROOT.exists(),
    reason="data/community/extracted is not present on this machine",
)


def test_splotopen_community_plot_satisfies_contract():
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    assert len(df) > 0
    assert validate_community_plot(df) == []


def test_splotopen_community_plot_row_count_matches_expected_scale():
    # 1,943,306 raw DT rows minus 1,678 with a null (unresolved) Species,
    # and 0 DT rows without a matching header plot in this dataset version
    # (verified directly against the extracted files).
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    assert len(df) == 1_941_628


def test_splotopen_community_plot_covers_every_header_plot():
    # Every one of the header table's 95,104 plots has at least one
    # resolved species row in this dataset version (verified directly).
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    assert df["plot_id"].nunique() == 95_104


def test_splotopen_community_plot_dataset_id_is_constant_splotopen():
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    assert (df["dataset_id"] == "sPlotOpen").all()


def test_splotopen_community_plot_preserves_source_database_provenance():
    df = load_splotopen_community_plot(COMMUNITY_ROOT)
    # manifest: 105 contributing source databases.
    assert df["source_database"].nunique() >= 100
    assert df["source_database"].isna().sum() == 0

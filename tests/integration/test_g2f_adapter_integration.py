"""Integration test: G2F adapter against the real data/external/g2f tables.

Skipped automatically if that data isn't present (e.g. a fresh checkout
without the SRG-GxE data symlink wired up) -- see README.md.
"""

from pathlib import Path

import pytest

from plant_context.data.contracts import (
    validate_environment_daily,
    validate_genotype_marker,
    validate_phenotype_plot,
)
from plant_context.data.g2f_adapter import (
    load_g2f_environment_daily,
    load_g2f_genotype_marker,
    load_g2f_phenotype_plot,
)

G2F_ROOT = Path(__file__).resolve().parents[2] / "data" / "external" / "g2f"

pytestmark = pytest.mark.skipif(
    not (G2F_ROOT / "phenotype.parquet").exists(),
    reason="data/external/g2f is not present on this machine",
)


def test_g2f_phenotype_plot_satisfies_contract():
    df = load_g2f_phenotype_plot(G2F_ROOT)
    assert len(df) > 0
    assert validate_phenotype_plot(df) == []


def test_g2f_genotype_marker_satisfies_contract():
    df = load_g2f_genotype_marker(G2F_ROOT)
    assert len(df) > 0
    assert validate_genotype_marker(df) == []


def test_g2f_environment_daily_satisfies_contract():
    df = load_g2f_environment_daily(G2F_ROOT)
    assert len(df) > 0
    assert validate_environment_daily(df) == []


def test_g2f_environment_daily_covers_most_environments():
    df = load_g2f_environment_daily(G2F_ROOT)
    # data_manifest.yaml notes daily weather covers 269 of 272 environments.
    assert df["environment_id"].nunique() >= 260

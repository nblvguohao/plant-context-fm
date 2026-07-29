"""Shared fixtures for the test suite."""

import pandas as pd
import pytest


@pytest.fixture
def synthetic_gxe_df() -> pd.DataFrame:
    """A small, fully-crossed genotype x environment x year fixture.

    6 genotypes x 6 environments x 4 years, every combination present, so
    there is enough redundancy for split functions (especially leave_ge) to
    have a genuine choice of valid folds rather than degenerating.
    """
    genotypes = [f"g{i}" for i in range(1, 7)]
    environments = [f"e{i}" for i in range(1, 7)]
    years = [2018, 2019, 2020, 2021]
    rows = [
        {"sample_id": f"{g}_{e}_{y}", "genotype_id": g, "environment_id": e, "year": y}
        for y in years
        for g in genotypes
        for e in environments
    ]
    return pd.DataFrame(rows)

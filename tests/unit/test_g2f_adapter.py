"""Unit tests for the G2F adapter (TDD 15 item 2), synthetic fixtures only.

Real-data checks against data/external/g2f live in
tests/integration/test_g2f_adapter_integration.py instead.
"""

import pandas as pd

from plant_context.data.contracts import (
    validate_environment_daily,
    validate_genotype_marker,
    validate_phenotype_plot,
)
from plant_context.data.g2f_adapter import (
    UNKNOWN_REFERENCE_BUILD,
    environment_daily_to_contract,
    genotype_to_contract,
    phenotype_to_contract,
)


def _raw_phenotype_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["DEH1_2014_g1_1_1", "DEH1_2014_g2_1_2"],
            "genotype_id": ["g1", "g2"],
            "environment_id": ["DEH1_2014", "DEH1_2014"],
            "year": [2014, 2014],
            "location_id": ["DEH1", "DEH1"],
            "trait_id": ["yield", "yield"],
            "trait_name": ["grain_yield", "grain_yield"],
            "trait_family": ["yield", "yield"],
            "phenotype_value": [5.72, 11.34],
            "phenotype_unit": ["Mg/ha", "Mg/ha"],
            "replicate_id": [1, 1],
            "block_id": [1, 1],
            "plot_id": [1, 2],
            "source_dataset": ["g2f_competition_2024", "g2f_competition_2024"],
            "grain_moisture": [20.8, 25.8],
        }
    )


def test_phenotype_to_contract_satisfies_contract():
    contract_df = phenotype_to_contract(_raw_phenotype_fixture())
    assert validate_phenotype_plot(contract_df) == []


def test_phenotype_to_contract_preserves_row_identity():
    contract_df = phenotype_to_contract(_raw_phenotype_fixture())
    assert contract_df["sample_id"].tolist() == ["DEH1_2014_g1_1_1", "DEH1_2014_g2_1_2"]
    assert contract_df["trait"].tolist() == ["yield", "yield"]
    assert contract_df["unit"].tolist() == ["Mg/ha", "Mg/ha"]
    assert contract_df["row"].isna().all()
    assert contract_df["column"].isna().all()


def _raw_genotype_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genotype_id": ["g1", "g1", "g2"],
            "marker_id": ["S1_100", "S1_200", "S1_100"],
            "chromosome": ["S1", "S1", "S1"],
            "position": [100, 200, 100],
            "allele_dosage": [0.0, 0.5, 1.0],
        }
    )


def test_genotype_to_contract_satisfies_contract_and_fills_reference_build():
    contract_df = genotype_to_contract(_raw_genotype_fixture())
    assert validate_genotype_marker(contract_df) == []
    assert (contract_df["reference_build"] == UNKNOWN_REFERENCE_BUILD).all()


def test_genotype_to_contract_accepts_explicit_reference_build():
    contract_df = genotype_to_contract(_raw_genotype_fixture(), reference_build="B73v5")
    assert (contract_df["reference_build"] == "B73v5").all()


def _raw_environment_fixture():
    weather_df = pd.DataFrame(
        {
            "environment_id": ["E1", "E1", "E1", "E1"],
            "date": pd.to_datetime(
                ["2020-04-10", "2020-04-14", "2020-04-15", "2020-04-16"]
            ),
            "tmax": [15.0, 20.0, 21.0, None],
            "tmin": [5.0, 8.0, 9.0, 9.5],
            "tmean": [10.0, 14.0, 15.0, 15.5],
            "precipitation": [0.0, 2.0, 0.0, 0.0],
            "solar_radiation": [10.0, 12.0, 13.0, 13.5],
            "relative_humidity": [70.0, 72.0, 71.0, 71.5],
        }
    )
    environment_df = pd.DataFrame(
        {
            "environment_id": ["E1"],
            "year": [2020],
            # day-of-year 106 = 2020-04-15 (2020 is a leap year)
            "estimated_planting_doy": [106],
        }
    )
    return weather_df, environment_df


def test_environment_daily_to_contract_satisfies_contract():
    weather_df, environment_df = _raw_environment_fixture()
    contract_df = environment_daily_to_contract(weather_df, environment_df)
    assert validate_environment_daily(contract_df) == []


def test_environment_daily_drops_rows_before_planting():
    weather_df, environment_df = _raw_environment_fixture()
    contract_df = environment_daily_to_contract(weather_df, environment_df)
    # 2020-04-10 and 2020-04-14 are before the 2020-04-15 planting date.
    assert contract_df["date"].min() == pd.Timestamp("2020-04-15")
    assert len(contract_df) == 2


def test_environment_daily_computes_days_after_planting():
    weather_df, environment_df = _raw_environment_fixture()
    contract_df = environment_daily_to_contract(weather_df, environment_df)
    row = contract_df[contract_df["date"] == pd.Timestamp("2020-04-16")].iloc[0]
    assert row["days_after_planting"] == 1


def test_environment_daily_flags_missing_weather_value():
    weather_df, environment_df = _raw_environment_fixture()
    contract_df = environment_daily_to_contract(weather_df, environment_df)
    row = contract_df[contract_df["date"] == pd.Timestamp("2020-04-16")].iloc[0]
    assert row["missing_flag"] == True  # noqa: E712 (tmax is None on this row)
    other_row = contract_df[contract_df["date"] == pd.Timestamp("2020-04-15")].iloc[0]
    assert other_row["missing_flag"] == False  # noqa: E712


def test_environment_daily_growth_stage_is_explicitly_unknown_not_guessed():
    weather_df, environment_df = _raw_environment_fixture()
    contract_df = environment_daily_to_contract(weather_df, environment_df)
    assert contract_df["growth_stage"].isna().all()

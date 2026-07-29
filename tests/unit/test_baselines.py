"""Unit tests for the ready-made crossfit baselines (TDD 15 item 4).

The GBLUP baseline has a real leakage subtlety worth pinning down as a
dedicated regression test: its allele-frequency centering must depend only
on a fold's train genotypes, never on what a held-out genotype's dosage
happens to be -- see gblup.py and baselines.py's docstrings for the bug
this used to have and how it was fixed.
"""

import numpy as np
import pandas as pd

from plant_context.statistics.baselines import make_gblup_predict_fn


def _synthetic_genotype_marker_df(genotype_dosages: dict) -> pd.DataFrame:
    rows = []
    for genotype_id, marker_values in genotype_dosages.items():
        for marker_id, (chromosome, position, dosage) in marker_values.items():
            rows.append(
                {
                    "genotype_id": genotype_id,
                    "marker_id": marker_id,
                    "chromosome": chromosome,
                    "position": position,
                    "allele_dosage": dosage,
                }
            )
    return pd.DataFrame(rows)


def test_gblup_predict_fn_predictions_do_not_depend_on_non_training_genotype_dosage():
    # g4 never appears in train_rows or eval_rows below -- only its raw
    # dosage differs between the two marker panels. Predictions for g3
    # (the genotype actually being evaluated) must be identical either way;
    # if the GRM's allele-frequency centering were (incorrectly) estimated
    # from all genotypes including g4, this would fail.
    genotypes_a = {
        "g1": {"m1": ("S1", 100, 0.0), "m2": ("S1", 200, 2.0)},
        "g2": {"m1": ("S1", 100, 1.0), "m2": ("S1", 200, 1.0)},
        "g3": {"m1": ("S1", 100, 2.0), "m2": ("S1", 200, 0.0)},
        "g4": {"m1": ("S1", 100, 0.0), "m2": ("S1", 200, 0.0)},
    }
    genotypes_b = {**genotypes_a, "g4": {"m1": ("S1", 100, 2.0), "m2": ("S1", 200, 2.0)}}

    df_a = _synthetic_genotype_marker_df(genotypes_a)
    df_b = _synthetic_genotype_marker_df(genotypes_b)

    train_rows = pd.DataFrame(
        {
            "genotype_id": ["g1", "g2"],
            "environment_id": ["e1", "e1"],
            "phenotype_value": [10.0, 20.0],
        }
    )
    eval_rows = pd.DataFrame({"genotype_id": ["g3"], "environment_id": ["e1"]})

    preds_a = make_gblup_predict_fn(df_a, max_dosage=2.0, n_folds=2)(train_rows, eval_rows)
    preds_b = make_gblup_predict_fn(df_b, max_dosage=2.0, n_folds=2)(train_rows, eval_rows)

    np.testing.assert_allclose(preds_a, preds_b)

"""Statistical test (TDD Section 10.4): 'known linear G-E should recover
effect direction'.

Simulates genotypes with known, deliberately varied sensitivity slopes
against an environment index, adds noise, fits the reaction norm, and
checks that the recovered slopes both correlate strongly with and share the
sign of the true slopes. This is a controlled-simulation test, not a
real-data test -- it exists to catch the reaction norm fit itself being
wrong, independent of any particular dataset.
"""

import numpy as np
import pandas as pd

from plant_context.evaluation.metrics import pearson_r
from plant_context.statistics.reaction_norm import fit_reaction_norm


def test_reaction_norm_recovers_slope_direction_under_noise():
    rng = np.random.default_rng(20260729)

    n_genotypes = 40
    true_intercepts = rng.uniform(4.0, 6.0, size=n_genotypes)
    true_slopes = rng.uniform(-3.0, 3.0, size=n_genotypes)  # spans both signs

    environment_index = pd.Series({f"e{i}": h for i, h in enumerate(np.linspace(-2, 2, 8))})

    rows = []
    for g_idx in range(n_genotypes):
        genotype_id = f"g{g_idx}"
        for env_id, h in environment_index.items():
            noise = rng.normal(scale=0.2)
            y = true_intercepts[g_idx] + true_slopes[g_idx] * h + noise
            rows.append(
                {"genotype_id": genotype_id, "environment_id": env_id, "phenotype_value": y}
            )
    train_df = pd.DataFrame(rows)

    fitted = fit_reaction_norm(train_df, environment_index)
    fitted_slopes = fitted.loc[[f"g{i}" for i in range(n_genotypes)], "b"].to_numpy()

    correlation = pearson_r(true_slopes, fitted_slopes)
    assert correlation > 0.95, f"expected strong slope recovery, got r={correlation}"

    sign_matches = np.sign(true_slopes) == np.sign(fitted_slopes)
    assert sign_matches.mean() > 0.9, "recovered slope direction disagrees with truth too often"

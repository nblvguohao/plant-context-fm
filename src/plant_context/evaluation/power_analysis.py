"""Statistical power analysis for model-comparison experiments (TDD Section 9,
Paper 3 target).

The SRG-GxE audit showed that n=3 seeds is far below the minimum detectable
effect (MDE) for typical G×E model gaps. This module provides:

- paired-difference MDE estimation;
- Nadeau-Bengio corrected resampled t-test (accounts for train/test overlap);
- seed-count recommendation given an expected effect size;
- a power report that flags underpowered comparisons in a results table.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def paired_mde(
    differences: Sequence[float],
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect size (absolute mean difference) for a paired
    two-sided t-test at significance level ``alpha`` and power ``power``.

    Uses the classical formula:
        MDE = (t_{1-alpha/2, df} + t_{power, df}) * SE_diff
    where SE_diff = sd(differences) / sqrt(n).
    """
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return float("nan")

    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1

    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_power = stats.t.ppf(power, df)
    return float((t_alpha + t_power) * se)


def nadeau_bengio_corrected_ttest(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> dict:
    """Nadeau-Bengio corrected resampled t-test for paired model comparison.

    When the same test set is reused across random seeds, standard paired
    t-tests are anti-conservative. Nadeau & Bengio (2003) inflate the
    variance by a factor that depends on the train/test ratio:

        var_corrected = var_diff * (1/n + n_test / n_train)

    where n is the number of paired scores (seeds).

    Parameters
    ----------
    scores_a, scores_b :
        Metric values (e.g. RMSE) for models A and B on the same test sets,
        one value per seed.
    n_train, n_test :
        Number of training and test samples used to produce the scores.
    alpha :
        Two-sided significance level.

    Returns
    -------
    dict with t_statistic, p_value, mean_diff, se_corrected, ci_lower, ci_upper,
    and corrected_mde.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    if n < 2:
        return {
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "mean_diff": float("nan"),
            "se_corrected": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "corrected_mde": float("nan"),
            "n_seeds": n,
        }

    diffs = a - b
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))

    # Nadeau-Bengio variance inflation
    inflation = 1.0 / n + n_test / n_train
    se_corrected = float(np.sqrt(var_diff * inflation))

    if se_corrected == 0:
        t_stat = float("inf") if mean_diff != 0 else 0.0
        p_value = 0.0 if mean_diff != 0 else 1.0
    else:
        t_stat = mean_diff / se_corrected
        df = n - 1
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    t_crit = stats.t.ppf(1 - alpha / 2, max(n - 1, 1))
    ci_lower = mean_diff - t_crit * se_corrected
    ci_upper = mean_diff + t_crit * se_corrected

    corrected_mde = paired_mde(diffs, alpha=alpha, power=0.8) * np.sqrt(inflation * n)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_diff": mean_diff,
        "se_corrected": se_corrected,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "corrected_mde": corrected_mde,
        "n_seeds": n,
    }


def recommend_seed_count(
    expected_effect: float,
    score_std: float,
    n_test: int,
    n_train: int,
    alpha: float = 0.05,
    power: float = 0.8,
    max_seeds: int = 20,
) -> dict:
    """Recommend the number of random seeds needed to detect ``expected_effect``.

    Uses the Nadeau-Bengio corrected standard error. Returns the smallest
    n_seeds such that the expected effect exceeds the corrected MDE.
    """
    if expected_effect <= 0 or score_std <= 0 or n_test <= 0 or n_train <= 0:
        return {
            "recommended_seeds": None,
            "achievable": False,
            "reason": "invalid inputs",
        }

    for n in range(2, max_seeds + 1):
        inflation = 1.0 / n + n_test / n_train
        se = score_std * np.sqrt(inflation)
        df = n - 1
        t_alpha = stats.t.ppf(1 - alpha / 2, df)
        t_power = stats.t.ppf(power, df)
        mde = (t_alpha + t_power) * se
        if expected_effect >= mde:
            return {
                "recommended_seeds": n,
                "achievable": True,
                "mde_at_n": float(mde),
            }

    n = max_seeds
    inflation = 1.0 / n + n_test / n_train
    se = score_std * np.sqrt(inflation)
    df = n - 1
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_power = stats.t.ppf(power, df)
    mde = (t_alpha + t_power) * se

    return {
        "recommended_seeds": max_seeds,
        "achievable": False,
        "mde_at_n": float(mde),
        "reason": f"effect smaller than MDE even with {max_seeds} seeds",
    }


def power_report_from_results(
    results_df: pd.DataFrame,
    split_col: str = "outer_split_type",
    model_col: str = "model",
    metric_col: str = "rmse",
    seed_col: str = "seed",
    n_train_col: str = "n_train",
    n_test_col: str = "n_test",
    baseline_model: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Flag underpowered model comparisons in a results table.

    For each split type, compare every non-baseline model to the baseline
    (or to every other model if no baseline is given) using the
    Nadeau-Bengio corrected t-test. Report whether the observed mean
    difference exceeds the corrected MDE.

    Expected columns in ``results_df``:
      - split_col, model_col, metric_col, seed_col
      - n_train_col, n_test_col (used for the correction)
    """
    if results_df.empty:
        return pd.DataFrame()

    rows = []
    for split_name, split_df in results_df.groupby(split_col):
        models = split_df[model_col].unique()
        if baseline_model is not None and baseline_model in models:
            comparisons = [(m, baseline_model) for m in models if m != baseline_model]
        else:
            comparisons = [
                (models[i], models[j])
                for i in range(len(models))
                for j in range(i + 1, len(models))
            ]

        for model_a, model_b in comparisons:
            a_df = split_df[split_df[model_col] == model_a].sort_values(seed_col)
            b_df = split_df[split_df[model_col] == model_b].sort_values(seed_col)
            if len(a_df) != len(b_df):
                continue

            n_train = int(a_df[n_train_col].iloc[0])
            n_test = int(a_df[n_test_col].iloc[0])

            test = nadeau_bengio_corrected_ttest(
                a_df[metric_col].to_numpy(),
                b_df[metric_col].to_numpy(),
                n_train=n_train,
                n_test=n_test,
                alpha=alpha,
            )

            rows.append(
                {
                    split_col: split_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_seeds": test["n_seeds"],
                    "mean_diff": test["mean_diff"],
                    "corrected_mde": test["corrected_mde"],
                    "p_value": test["p_value"],
                    "ci_lower": test["ci_lower"],
                    "ci_upper": test["ci_upper"],
                    "powered": abs(test["mean_diff"]) > test["corrected_mde"],
                }
            )

    return pd.DataFrame(rows)

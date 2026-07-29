"""Simple low-rank G-E model (TDD Section 6.4, 15 item 7).

Predicts phenotype directly from genotype LD-block tokens
(tokenizers/genotype.py) and environment growth-stage tokens
(tokenizers/environment.py), via the simplest of TDD 6.4's three fusion
candidates: a low-rank bilinear interaction. TDD 6.4's own stated principle
is to start with the simplest model and only add complexity once a paired
ablation demonstrates a real gain -- FiLM and cross-attention fusion are
deliberately not implemented here, and this model does not yet wrap a
statistical main-effects baseline as a residual target (TDD 6.3's full
decomposition); that residual-learning framing is TDD 15 item 8, which
audits/extends the existing SRG-GxE codebase instead of building a second,
parallel implementation of the same idea.

    y_hat(g, e) = global_mean + genotype_main(g) + environment_main(e)
                  + <genotype_embed(g), environment_embed(e)>   [if use_interaction]

genotype_main/environment_main/genotype_embed/environment_embed are all
small linear(+ReLU) heads over the tokenized *features*, not a learned
per-ID embedding table indexed by genotype_id/environment_id -- so an
unseen genotype or environment at prediction time still gets a prediction
from its own marker/weather features, the same generalization property
that makes GBLUP work under leave-genotype-out.

Standardization of the input features is a train-only statistic, same as
GBLUP's allele-frequency centering (see gblup.py's docstring for the bug
that pattern caught there): ``make_low_rank_gxe_predict_fn``'s per-fold
closure computes mean/std from that fold's train genotypes/environments
only, then applies it to the full feature tables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from plant_context.statistics.crossfit import FitPredictFn


def pivot_genotype_tokens_wide(genotype_block_tokens: pd.DataFrame) -> pd.DataFrame:
    """(genotype_id, ld_block_id, mean_dosage) long tokens -> genotype x block wide matrix.

    A genotype missing a given block (including the OOV block, if any
    marker was unseen at LD-block-fit time) gets that column's mean dosage
    as an explicit, documented fallback rather than an arbitrary zero.
    """
    wide = genotype_block_tokens.pivot_table(
        index="genotype_id", columns="ld_block_id", values="mean_dosage", aggfunc="first"
    )
    if wide.isna().any().any():
        wide = wide.fillna(wide.mean(axis=0))
    return wide


def pivot_environment_tokens_wide(environment_stage_tokens: pd.DataFrame) -> pd.DataFrame:
    """(environment_id, growth_stage, ...features) long tokens -> environment x flattened-feature wide matrix."""
    feature_cols = [
        c
        for c in environment_stage_tokens.columns
        if c not in ("environment_id", "growth_stage", "stage_estimation_method")
    ]
    wide = environment_stage_tokens.pivot_table(
        index="environment_id", columns="growth_stage", values=feature_cols, aggfunc="first"
    )
    wide.columns = [f"{stage}__{feature}" for feature, stage in wide.columns]
    if wide.isna().any().any():
        wide = wide.fillna(wide.mean(axis=0))
    return wide


class LowRankGxEModel(nn.Module):
    def __init__(
        self,
        n_genotype_features: int,
        n_environment_features: int,
        rank: int = 8,
        hidden: int = 32,
        use_interaction: bool = True,
    ):
        super().__init__()
        self.use_interaction = use_interaction
        self.global_mean = nn.Parameter(torch.zeros(1))
        self.genotype_main = nn.Sequential(
            nn.Linear(n_genotype_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.environment_main = nn.Sequential(
            nn.Linear(n_environment_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        if use_interaction:
            self.genotype_embed = nn.Linear(n_genotype_features, rank)
            self.environment_embed = nn.Linear(n_environment_features, rank)

    def forward(
        self, genotype_features: torch.Tensor, environment_features: torch.Tensor
    ) -> torch.Tensor:
        g_main = self.genotype_main(genotype_features).squeeze(-1)
        e_main = self.environment_main(environment_features).squeeze(-1)
        out = self.global_mean + g_main + e_main
        if self.use_interaction:
            u = self.genotype_embed(genotype_features)
            v = self.environment_embed(environment_features)
            out = out + (u * v).sum(dim=-1)
        return out


def _require_present(ids: pd.Series, features: pd.DataFrame, label: str) -> None:
    missing = set(ids) - set(features.index)
    if missing:
        raise ValueError(
            f"{label} ids {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            f"are not present in the provided feature table -- prepare features for "
            f"every {label} id used in train/eval rows before calling this function"
        )


def make_low_rank_gxe_predict_fn(
    genotype_features: pd.DataFrame,
    environment_features: pd.DataFrame,
    rank: int = 8,
    hidden: int = 32,
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 1234,
    use_interaction: bool = True,
    device: str = "cpu",
) -> FitPredictFn:
    """Build a low-rank G-E fit_predict_fn for use with run_crossfit.

    Standardization statistics (mean/std of each feature column) are
    computed fresh inside the returned closure, restricted to that fold's
    train genotypes/environments, then applied to the full feature tables
    -- never fit using an outer-test genotype or environment's own data.
    """

    def _fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        _require_present(train_rows["genotype_id"], genotype_features, "genotype")
        _require_present(eval_rows["genotype_id"], genotype_features, "genotype")
        _require_present(train_rows["environment_id"], environment_features, "environment")
        _require_present(eval_rows["environment_id"], environment_features, "environment")

        torch.manual_seed(seed)

        train_genotype_ids = set(train_rows["genotype_id"])
        train_environment_ids = set(train_rows["environment_id"])
        g_train_raw = genotype_features.loc[genotype_features.index.isin(train_genotype_ids)]
        e_train_raw = environment_features.loc[environment_features.index.isin(train_environment_ids)]

        g_mean, g_std = g_train_raw.mean(axis=0), g_train_raw.std(axis=0).replace(0, 1.0)
        e_mean, e_std = e_train_raw.mean(axis=0), e_train_raw.std(axis=0).replace(0, 1.0)
        g_scaled = (genotype_features - g_mean) / g_std
        e_scaled = (environment_features - e_mean) / e_std

        model = LowRankGxEModel(
            n_genotype_features=g_scaled.shape[1],
            n_environment_features=e_scaled.shape[1],
            rank=rank,
            hidden=hidden,
            use_interaction=use_interaction,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        g_tensor_train = torch.tensor(
            g_scaled.reindex(train_rows["genotype_id"]).to_numpy(), dtype=torch.float32, device=device
        )
        e_tensor_train = torch.tensor(
            e_scaled.reindex(train_rows["environment_id"]).to_numpy(), dtype=torch.float32, device=device
        )
        y_train = torch.tensor(
            train_rows["phenotype_value"].to_numpy(), dtype=torch.float32, device=device
        )

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(g_tensor_train, e_tensor_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            g_tensor_eval = torch.tensor(
                g_scaled.reindex(eval_rows["genotype_id"]).to_numpy(), dtype=torch.float32, device=device
            )
            e_tensor_eval = torch.tensor(
                e_scaled.reindex(eval_rows["environment_id"]).to_numpy(), dtype=torch.float32, device=device
            )
            predictions = model(g_tensor_eval, e_tensor_eval).cpu().numpy()
        return predictions

    return _fit_predict

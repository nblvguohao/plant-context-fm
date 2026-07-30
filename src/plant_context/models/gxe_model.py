"""Low-rank G-E models (TDD Section 6.4, 15 item 7, Gate D).

Two architectures:

1. ``LowRankGxEModel`` — the original (TDD 6.4): takes wide-format flat
   genotype features (genotype × block-dosage matrix) and environment
   features.  Simple and fast.

2. ``EncoderGxEModel`` (Gate D): takes genotype BLOCK SEQUENCES with an
   optional pretrained ``GenotypeBlockEncoder``.  The encoder processes the
   per-genotype ordered block sequence and produces a pooled embedding that
   replaces the wide feature vector.  This allows evaluating whether masked-
   reconstruction pretraining on block sequences improves G×E prediction.

Both predict phenotype from genotype and environment features via the same
bilinear decomposition (the second just replaces the genotype input stage):

    y_hat(g, e) = global_mean + genotype_main(g) + environment_main(e)
                  + <genotype_embed(g), environment_embed(e)>   [if use_interaction]

genotype_main/environment_main/genotype_embed/environment_embed are all
small linear(+ReLU) heads over the tokenized *features*, not a learned
per-ID embedding table indexed by genotype_id/environment_id — so an
unseen genotype or environment at prediction time still gets a prediction
from its own marker/weather features, the same generalization property
that makes GBLUP work under leave-genotype-out.

Standardization of the input features is a train-only statistic, same as
GBLUP's allele-frequency centering.
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


# ── Encoder-based G×E (Gate D) ─────────────────────────────────────────────


class EncoderGxEModel(nn.Module):
    """G×E model with a ``TokenSequenceEncoder`` for genotype block sequences.

    Instead of wide-format flat features, this takes ordered block tokens per
    genotype, encodes them into ``d_model``-dim pooled embeddings via the
    encoder, and feeds those to the same bilinear heads as
    ``LowRankGxEModel``.

    The encoder can be initialised from a pretrained ``GenotypeBlockEncoder``
    (or a randomly-initialised ``TokenSequenceEncoder`` for the scratch
    condition), enabling Gate D's comparison.

    Parameters
    ----------
    n_block_features :
        Number of features per LD-block token (e.g. 1 for mean_dosage only).
    n_environment_features :
        Number of wide-format environment features.
    d_model :
        Transformer embedding dimension (= genotype input dim for the heads).
    rank, hidden, use_interaction :
        Same as ``LowRankGxEModel``.
    genotype_encoder :
        Optional pre-initialised ``TokenSequenceEncoder`` or ``None`` for a
        fresh randomly-initialised one.
    """

    def __init__(
        self,
        n_block_features: int,
        n_environment_features: int,
        d_model: int = 32,
        rank: int = 8,
        hidden: int = 32,
        use_interaction: bool = True,
        genotype_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.use_interaction = use_interaction
        self.global_mean = nn.Parameter(torch.zeros(1))

        if genotype_encoder is not None:
            self.genotype_encoder = genotype_encoder
        else:
            from plant_context.models.context_encoder import TokenSequenceEncoder
            self.genotype_encoder = TokenSequenceEncoder(
                n_features=n_block_features, d_model=d_model,
            )

        self.environment_main = nn.Sequential(
            nn.Linear(n_environment_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.genotype_main = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        if use_interaction:
            self.genotype_embed = nn.Linear(d_model, rank)
            self.environment_embed = nn.Linear(n_environment_features, rank)

    def forward(
        self,
        genotype_tokens: torch.Tensor,
        environment_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        genotype_tokens :
            ``[batch, n_blocks, n_block_features]`` — the ordered block tokens.
        environment_features :
            ``[batch, n_env_features]`` — wide-format environment features.
        padding_mask :
            ``[batch, n_blocks]`` — True = block absent.

        Returns
        -------
        ``[batch]`` — predicted phenotype values.
        """
        _, g_pooled = self.genotype_encoder(genotype_tokens, key_padding_mask=padding_mask)
        g_main = self.genotype_main(g_pooled).squeeze(-1)
        e_main = self.environment_main(environment_features).squeeze(-1)
        out = self.global_mean + g_main + e_main
        if self.use_interaction:
            u = self.genotype_embed(g_pooled)
            v = self.environment_embed(environment_features)
            out = out + (u * v).sum(dim=-1)
        return out


class MLPGxEModel(nn.Module):
    """G×E model with per-block MLP + pooling (NO cross-block attention).

    Ablation for Gate D: same block-sequence input as EncoderGxEModel,
    similar parameter count (~10K), but processes each block independently
    with a per-block MLP and then mean-pools across blocks. This isolates
    whether the Transformer's self-attention (cross-block communication)
    provides value beyond per-block processing.

    If MLPGxEModel performs as well as EncoderGxEModel, the advantage is
    from per-block processing, not from sequence-aware attention. If it
    performs worse, the Transformer's ability to share information across
    blocks is the source of improvement.
    """

    def __init__(
        self,
        n_block_features: int,
        n_environment_features: int,
        d_model: int = 32,
        mlp_hidden: int = 128,
        rank: int = 8,
        hidden: int = 32,
        use_interaction: bool = True,
    ):
        super().__init__()
        self.use_interaction = use_interaction
        self.global_mean = nn.Parameter(torch.zeros(1))

        # Per-block MLP (no cross-block communication)
        self.per_block_mlp = nn.Sequential(
            nn.Linear(n_block_features, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, d_model), nn.ReLU(),
        )

        self.environment_main = nn.Sequential(
            nn.Linear(n_environment_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.genotype_main = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        if use_interaction:
            self.genotype_embed = nn.Linear(d_model, rank)
            self.environment_embed = nn.Linear(n_environment_features, rank)

    def forward(
        self,
        genotype_tokens: torch.Tensor,
        environment_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — per-block MLP then mean pooling."""
        # [batch, n_blocks, n_features] → [batch, n_blocks, d_model]
        g = self.per_block_mlp(genotype_tokens)
        # Mean pool over non-masked blocks
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            g_pooled = (g * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            g_pooled = g.mean(dim=1)

        g_main = self.genotype_main(g_pooled).squeeze(-1)
        e_main = self.environment_main(environment_features).squeeze(-1)
        out = self.global_mean + g_main + e_main
        if self.use_interaction:
            u = self.genotype_embed(g_pooled)
            v = self.environment_embed(environment_features)
            out = out + (u * v).sum(dim=-1)
        return out


def make_mlp_gxe_predict_fn(
    per_genotype_tokens: dict,
    environment_features: pd.DataFrame,
    block_feature_columns: list[str] = ("mean_dosage",),
    d_model: int = 32,
    mlp_hidden: int = 128,
    rank: int = 8,
    hidden: int = 32,
    epochs: int = 300,
    lr: float = 0.001,
    seed: int = 1234,
    use_interaction: bool = True,
    device: str = "cpu",
    batch_size: Optional[int] = None,
) -> FitPredictFn:
    """Build an MLP-ablation G×E predict function for crossfit.

    Same interface as ``make_encoder_gxe_predict_fn`` but uses
    ``MLPGxEModel`` instead of ``EncoderGxEModel``.

    Parameters
    ----------
    batch_size :
        If set, training uses mini-batches of this many samples to reduce
        GPU memory usage.  Useful for large datasets (e.g. full-scale 4938
        genotypes).  If None, processes all samples in a single batch.
    """
    n_block_features = len(block_feature_columns)
    n_env_features = environment_features.shape[1]

    def _fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        torch.manual_seed(seed)

        train_genotype_ids = set(train_rows["genotype_id"])
        train_environment_ids = set(train_rows["environment_id"])

        e_train_raw = environment_features.loc[
            environment_features.index.isin(train_environment_ids)
        ]
        e_mean, e_std = e_train_raw.mean(axis=0), e_train_raw.std(axis=0).replace(0, 1.0)
        e_scaled = (environment_features - e_mean) / e_std

        g_tensor_all, g_mask_all, g_ids = build_block_tensor(
            per_genotype_tokens, block_feature_columns
        )
        train_mask = torch.tensor(
            [sid in train_genotype_ids for sid in g_ids], dtype=torch.bool
        )
        g_tensor_all = torch.nan_to_num(g_tensor_all, nan=0.0)
        train_features = g_tensor_all[train_mask][~g_mask_all[train_mask]]
        g_mean = train_features.mean(dim=0)
        g_std = train_features.std(dim=0).clamp(min=1e-8)
        g_scaled = (g_tensor_all - g_mean) / g_std
        g_scaled[g_mask_all] = 0.0

        model = MLPGxEModel(
            n_block_features=n_block_features,
            n_environment_features=n_env_features,
            d_model=d_model,
            mlp_hidden=mlp_hidden,
            rank=rank,
            hidden=hidden,
            use_interaction=use_interaction,
        ).to(device)

        gid_to_idx = {gid: i for i, gid in enumerate(g_ids)}

        def _tensors(rows):
            g_idx = torch.tensor(
                [gid_to_idx[gid] for gid in rows["genotype_id"]], dtype=torch.long
            )
            e_t = torch.tensor(
                e_scaled.reindex(rows["environment_id"]).to_numpy(),
                dtype=torch.float32, device=device,
            )
            return g_idx, e_t

        g_train_idx, e_train_t = _tensors(train_rows)
        g_eval_idx, e_eval_t = _tensors(eval_rows)
        y_train = torch.tensor(
            train_rows["phenotype_value"].to_numpy(), dtype=torch.float32, device=device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_train = len(train_rows)

        model.train()
        for _ in range(epochs):
            if batch_size is not None and n_train > batch_size:
                # Mini-batch training (each batch gets its own optimizer step)
                epoch_perm = torch.randperm(n_train)
                for start in range(0, n_train, batch_size):
                    batch_idx = epoch_perm[start:start + batch_size]
                    optimizer.zero_grad()
                    g_idx = g_train_idx[batch_idx]
                    preds = model(
                        g_scaled[g_idx].to(device),
                        e_train_t[batch_idx.to(device)],
                        padding_mask=g_mask_all[g_idx].to(device),
                    )
                    loss = loss_fn(preds, y_train[batch_idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                optimizer.zero_grad()
                preds = model(
                    g_scaled[g_train_idx].to(device), e_train_t,
                    padding_mask=g_mask_all[g_train_idx].to(device),
                )
                loss = loss_fn(preds, y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(
                g_scaled[g_eval_idx].to(device), e_eval_t,
                padding_mask=g_mask_all[g_eval_idx].to(device),
            ).cpu().numpy()
        return predictions

    return _fit_predict


def build_block_tensor(
    per_genotype_tokens: dict,
    feature_columns: list,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Build padded ``[n_genotypes, max_blocks, n_features]`` tensor + mask.

    Parameters
    ----------
    per_genotype_tokens :
        ``{genotype_id: DataFrame indexed by ld_block_id,
        columns = feature_columns}``.
    feature_columns :
        Which columns to use.

    Returns
    -------
    (tensor [batch, max_blocks, n_features], mask [batch, max_blocks], genotype_ids)
    """
    import numpy as np
    sample_ids = list(per_genotype_tokens.keys())
    max_len = max(len(df) for df in per_genotype_tokens.values()) if sample_ids else 0
    n_features = len(feature_columns)

    data = np.zeros((len(sample_ids), max_len, n_features), dtype=np.float32)
    mask = np.ones((len(sample_ids), max_len), dtype=bool)

    for i, gid in enumerate(sample_ids):
        df = per_genotype_tokens[gid]
        n = len(df)
        data[i, :n, :] = df[feature_columns].to_numpy(dtype=np.float32)
        mask[i, :n] = False

    return torch.tensor(data), torch.tensor(mask), sample_ids


def make_encoder_gxe_predict_fn(
    per_genotype_tokens: dict,
    environment_features: pd.DataFrame,
    block_feature_columns: list[str] = ("mean_dosage",),
    d_model: int = 32,
    rank: int = 8,
    hidden: int = 32,
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 1234,
    use_interaction: bool = True,
    device: str = "cpu",
    genotype_encoder: Optional[nn.Module] = None,
) -> FitPredictFn:
    """Build an encoder-based G×E predict function for crossfit.

    Parameters
    ----------
    per_genotype_tokens :
        ``{genotype_id: DataFrame indexed by ld_block_id,
        columns = feature_columns}``.
    environment_features :
        Wide-format DataFrame indexed by environment_id.
    block_feature_columns :
        Which block token columns are features.
    genotype_encoder :
        Optional pretrained encoder to initialise with.  If None, a fresh
        randomly-initialised encoder is created (= scratch condition).

    All other parameters match ``make_low_rank_gxe_predict_fn``.
    """
    n_block_features = len(block_feature_columns)
    n_env_features = environment_features.shape[1]

    def _fit_predict(train_rows: pd.DataFrame, eval_rows: pd.DataFrame) -> np.ndarray:
        torch.manual_seed(seed)

        train_genotype_ids = set(train_rows["genotype_id"])
        train_environment_ids = set(train_rows["environment_id"])

        # Standardise environment features (train-only stats)
        e_train_raw = environment_features.loc[
            environment_features.index.isin(train_environment_ids)
        ]
        e_mean, e_std = e_train_raw.mean(axis=0), e_train_raw.std(axis=0).replace(0, 1.0)
        e_scaled = (environment_features - e_mean) / e_std

        # Build block tensors for train and eval
        g_tensor_all, g_mask_all, g_ids = build_block_tensor(
            per_genotype_tokens, block_feature_columns
        )

        # Standardise block features (train-only stats)
        train_mask = torch.tensor(
            [sid in train_genotype_ids for sid in g_ids], dtype=torch.bool
        )
        # Replace NaN with 0.0 first (blocks where a genotype has no valid dosage)
        g_tensor_all = torch.nan_to_num(g_tensor_all, nan=0.0)
        train_features = g_tensor_all[train_mask][~g_mask_all[train_mask]]
        g_mean = train_features.mean(dim=0)
        g_std = train_features.std(dim=0).clamp(min=1e-8)
        g_scaled = (g_tensor_all - g_mean) / g_std
        g_scaled[g_mask_all] = 0.0  # masked positions get 0 after standardisation

        # Build model (no pretrained encoder — load weights below)
        model = EncoderGxEModel(
            n_block_features=n_block_features,
            n_environment_features=n_env_features,
            d_model=d_model,
            rank=rank,
            hidden=hidden,
            use_interaction=use_interaction,
            genotype_encoder=None,
        ).to(device)

        # Copy pretrained encoder weights into the fresh model (clone state
        # dict so each fold gets its own copy, avoiding cross-fold bleed)
        if genotype_encoder is not None:
            import copy
            model.genotype_encoder.load_state_dict(
                copy.deepcopy(genotype_encoder.state_dict())
            )

        # Map genotype IDs to tensor indices
        gid_to_idx = {gid: i for i, gid in enumerate(g_ids)}

        def _tensors(rows):
            g_idx = torch.tensor(
                [gid_to_idx[gid] for gid in rows["genotype_id"]], dtype=torch.long
            )
            e_t = torch.tensor(
                e_scaled.reindex(rows["environment_id"]).to_numpy(),
                dtype=torch.float32, device=device,
            )
            return g_idx, e_t

        g_train_idx, e_train_t = _tensors(train_rows)
        g_eval_idx, e_eval_t = _tensors(eval_rows)
        y_train = torch.tensor(
            train_rows["phenotype_value"].to_numpy(), dtype=torch.float32, device=device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(
                g_scaled[g_train_idx].to(device),
                e_train_t,
                padding_mask=g_mask_all[g_train_idx].to(device),
            )
            loss = loss_fn(preds, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(
                g_scaled[g_eval_idx].to(device),
                e_eval_t,
                padding_mask=g_mask_all[g_eval_idx].to(device),
            ).cpu().numpy()
        return predictions

    return _fit_predict

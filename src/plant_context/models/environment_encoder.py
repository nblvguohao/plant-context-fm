"""Shared environment encoder (TDD Section 6.1, 15 item 12).

The environment representation is the piece meant to be *shared* between the
community ecology and G×E prediction branches (TDD Section 2: "only the
environment representation is meant to be shared").  The core Transformer
architecture is ``TokenSequenceEncoder`` (context_encoder.py) — this module
wraps it with environment-specific I/O: padding, stage-order alignment,
missing-stage masking, and utilities for transfer between weather-derived and
community-derived environment features.

Two training regimes are supported (TDD 6.1: "can be frozen, fine-tuned, or
trained from scratch"):

1. **Pretrain on community-derived features**: treat each environment's genus-
   incidence or abundance-rank vector as the per-"stage" token sequence and
   run masked reconstruction (TDD 7.3 / item 9's pretraining loop).

2. **Fine-tune on G×E weather features**: initialise from the community-
   pretrained checkpoint, then train end-to-end in the low-rank G×E model
   (gxe_model.py).

Bridge experiment: compare G×E prediction accuracy with and without community-
pretrained encoder initialisation to quantify the value transfer across the
two domains (TDD 14 risk table "bridge experiments").
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from plant_context.models.context_encoder import (
    TokenSequenceEncoder,
    check_collapse_gate,
    embedding_collapse_diagnostics,
)
from plant_context.models.pretraining import pretrain_masked_reconstruction
from plant_context.tokenizers.masking import mask_contiguous_run


class SharedEnvironmentEncoder(nn.Module):
    """Environment encoder shared between community and G×E branches.

    Wraps ``TokenSequenceEncoder`` with environment-specific data handling:
    each environment is a sequence of "stages" (weather phenological stages,
    or genus/species groups for community data), and a missing stage mask is
    carried through the encoder to avoid treating absent stages as zero.

    Parameters
    ----------
    n_stage_features :
        Number of feature values per stage token (e.g. 8 weather metrics in
        EnvironmentTokenizer, or incidence of N genera from the community
        bridge).
    d_model, n_heads, n_layers :
        Transformer architecture — kept small per TDD 6.1's "controlled
        parameter count" requirement.
    stage_names :
        Optional ordered list of stage names (e.g. ``STAGE_ORDER`` from
        environment.py).  Used for input validation and deterministic column
        ordering.
    """

    def __init__(
        self,
        n_stage_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        stage_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.encoder = TokenSequenceEncoder(
            n_features=n_stage_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.stage_names = list(stage_names) if stage_names is not None else []
        self._d_model = d_model
        self._n_layers = n_layers
        self._n_heads = n_heads

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_heads(self) -> int:
        return self._n_heads

    def forward(
        self,
        stage_features: torch.Tensor,
        stage_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of environment stage sequences.

        Parameters
        ----------
        stage_features :
            ``[batch, n_stages, n_stage_features]`` tensor, already aligned
            to a fixed stage order (missing stages zero-filled).
        stage_mask :
            ``[batch, n_stages]`` bool, True = this stage is absent/should be
            masked from attention and pooling.  If None, all stages are
            assumed present.

        Returns
        -------
        stage_embeddings : ``[batch, n_stages, d_model]`` — per-stage tokens.
        pooled : ``[batch, d_model]`` — mean over non-masked stages.
        """
        return self.encoder(stage_features, key_padding_mask=stage_mask)

    @torch.no_grad()
    def embed_dataframe(
        self,
        environment_features: pd.DataFrame,
        stage_order: Sequence[str],
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Convert a per-environment stage-feature table into pooled
        environment embeddings.

        Parameters
        ----------
        environment_features :
            DataFrame indexed by ``environment_id``, columns are
            ``{stage_order}__{feature_columns}`` (the wide format produced by
            ``pivot_environment_tokens_wide`` or by the community bridge's
            ``aggregate_community_features`` when a stage-like reindex is
            applied).
        stage_order :
            Ordered list of stage identifiers, e.g. ``STAGE_ORDER``.
        feature_columns :
            Ordered list of feature metric names per stage.

        Returns
        -------
        DataFrame indexed by ``environment_id`` with ``d_model`` embedding
        columns named ``embed_0`` … ``embed_{d_model-1}``.
        """
        tensor, mask = self._build_tensor(environment_features, stage_order, feature_columns)
        self.eval()
        with torch.no_grad():
            _, pooled = self.forward(tensor, stage_mask=mask)

        env_ids = environment_features.index
        return pd.DataFrame(
            pooled.numpy(),
            index=env_ids,
            columns=[f"embed_{i}" for i in range(self.d_model)],
        )

    def _build_tensor(
        self,
        environment_features: pd.DataFrame,
        stage_order: Sequence[str],
        feature_columns: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a padded ``[n_envs, n_stages, n_features]`` tensor and a
        boolean mask from a wide-format environment feature table.

        For each stage in ``stage_order``, we look for the expected columns
        ``{stage}__{feature}``.  A stage that has no columns in the DataFrame
        for a given environment is marked as masked.
        """
        n_envs = len(environment_features)
        n_stages = len(stage_order)
        n_features = len(feature_columns)

        data = np.zeros((n_envs, n_stages, n_features), dtype=np.float32)
        mask = np.ones((n_envs, n_stages), dtype=bool)

        for s_idx, stage in enumerate(stage_order):
            cols = [f"{stage}__{f}" for f in feature_columns]
            present = [c for c in cols if c in environment_features.columns]
            if not present:
                continue  # entire stage masked
            # Subset to columns that are actually present
            stage_data = environment_features[present].to_numpy(dtype=np.float32)
            # Map to the full feature column order — missing sub-features
            # within a present stage are filled with NaN (to be masked per env)
            full = np.full((n_envs, n_features), np.nan, dtype=np.float32)
            for f_idx, f_name in enumerate(feature_columns):
                col = f"{stage}__{f_name}"
                if col in environment_features.columns:
                    full[:, f_idx] = stage_data[:, present.index(col)]

            # An environment is masked for this stage if ALL its features are NaN
            row_nan = np.isnan(full).all(axis=1)
            full[row_nan] = 0.0
            mask[:, s_idx] = row_nan
            data[:, s_idx, :] = full

        return torch.tensor(data), torch.tensor(mask)


def pretrain_environment_encoder(
    environment_features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
    mask_fraction: float = 0.2,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 1234,
    d_model: int = 32,
    encoder: Optional[SharedEnvironmentEncoder] = None,
    device: str = "cpu",
) -> dict:
    """Pretrain a ``SharedEnvironmentEncoder`` via masked stage
    reconstruction on environment-stage data (weather or community).

    Each environment's stage sequence is treated as a sample; stages are
    masked with a contiguous run (``mask_contiguous_run``) so that the
    encoder must learn inter-stage structure.

    ``encoder`` may be a pre-initialised ``SharedEnvironmentEncoder`` for
    fine-tuning from a pretrained checkpoint. If None, a fresh encoder is
    created.

    Returns a dict with ``encoder`` (SharedEnvironmentEncoder), ``head``,
    ``loss_history``, ``final_loss``, ``collapse_diagnostics``.
    """
    n_stage_features = len(feature_columns)

    # Seed before creating the encoder so that a freshly-built encoder has
    # deterministic initial weights. ``pretrain_masked_reconstruction`` also
    # seeds internally for mask/dropout determinism.
    torch.manual_seed(seed)

    # Build per-environment tensor internally
    if encoder is None:
        enc = SharedEnvironmentEncoder(
            n_stage_features=n_stage_features,
            d_model=d_model,
            stage_names=stage_order,
        )
    else:
        enc = encoder

    tensor, mask = enc._build_tensor(environment_features, stage_order, feature_columns)

    # Pack into the per-sample-tokens format pretrain_masked_reconstruction
    # expects: {sample_id: DataFrame indexed by stage, columns = feature_columns}
    per_sample = {}
    env_ids = environment_features.index
    for i, env_id in enumerate(env_ids):
        present_stages = [stage_order[s] for s in range(len(stage_order)) if not mask[i, s]]
        if len(present_stages) < 2:
            continue  # need at least 2 stages for contiguous masking
        stage_df = pd.DataFrame(
            tensor[i, ~mask[i], :].numpy(),
            index=present_stages,
            columns=feature_columns,
        )
        per_sample[env_id] = stage_df

    mask_fn = MaskPartial(mask_contiguous_run, mask_fraction=mask_fraction)

    result = pretrain_masked_reconstruction(
        per_sample_tokens=per_sample,
        feature_columns=list(feature_columns),
        mask_fn=mask_fn,
        d_model=d_model,
        epochs=epochs,
        lr=lr,
        seed=seed,
        encoder=enc.encoder,
        device=device,
    )

    # Replace the generic encoder with a SharedEnvironmentEncoder wrapping
    # the trained weights
    enc.encoder.load_state_dict(result["encoder"].state_dict())
    result["encoder"] = enc

    return result


class MaskPartial:
    """Wrapper so ``functools.partial(mask_contiguous_run, mask_fraction=0.2)``
    can pass ``seed`` as keyword without colliding with the bound ``mask_fraction``.

    ``pretrain_masked_reconstruction`` calls ``mask_fn(token_ids, seed=seed + i)``.
    """

    def __init__(self, fn: Callable, **bound_kwargs):
        self._fn = fn
        self._bound = bound_kwargs

    def __call__(self, token_ids, **kwargs):
        merged = dict(self._bound)
        merged.update(kwargs)
        return self._fn(token_ids, **merged)


def _validate_feature_dimensions(
    weather_features: pd.DataFrame,
    community_features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
) -> bool:
    """Check that weather and community features have the same per-stage
    feature dimensionality after bridge aggregation.  This is a necessary
    condition for weight sharing (same n_stage_features → same encoder).
    """
    n_weather = len(feature_columns)

    # Community features may not have stage_order columns; we check the
    # per-observation feature dimension instead.
    n_community = None
    for stage in stage_order:
        cols = [f"{stage}__{f}" for f in feature_columns]
        present = [c for c in cols if c in community_features.columns]
        if present:
            n_community = n_community or len(present)
            if len(present) != n_weather:
                return False

    # If no community columns with the expected pattern, check raw feature count
    if n_community is None:
        n_community = community_features.shape[1]

    return n_weather == n_community


def bridge_transfer_experiment(
    weather_features: pd.DataFrame,
    community_features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
    pretrain_epochs: int = 30,
    finetune_epochs: int = 50,
    seed: int = 1234,
    device: str = "cpu",
) -> dict:
    """Run a bridge transfer experiment: pretrain on community-derived
    features → fine-tune on weather-derived features.

    This is the experiment that tests whether community ecology data provides
    a useful initialisation for G×E environment encoding (TDD 14's "bridge
    experiments" from the risk table, showing the two tracks are not
    disconnected).

    Parameters
    ----------
    weather_features, community_features :
        Wide-format DataFrames indexed by environment_id (only overlapping
        environment IDs across both tables are used).
    stage_order, feature_columns :
        Stage and feature ordering for the shared encoder.
    pretrain_epochs :
        Masked-reconstruction epochs on community features.
    finetune_epochs :
        Masked-reconstruction epochs on weather features (from community init).

    Returns a dict with ``community_pretrain``, ``weather_finetune`` results,
    and a ``shared_environments`` count.
    """
    common_envs = weather_features.index.intersection(community_features.index)
    if len(common_envs) == 0:
        return {
            "shared_environments": 0,
            "community_pretrain": None,
            "weather_finetune": None,
            "status": "no_shared_environments",
        }

    weather_sub = weather_features.loc[common_envs]
    community_sub = community_features.loc[common_envs]

    # Phase 1: pretrain on community features
    community_result = pretrain_environment_encoder(
        community_sub,
        stage_order=stage_order,
        feature_columns=feature_columns,
        epochs=pretrain_epochs,
        lr=0.01,
        seed=seed,
        device=device,
    )

    # Phase 2: fine-tune on weather features starting from community weights
    finetune_result = pretrain_environment_encoder(
        weather_sub,
        stage_order=stage_order,
        feature_columns=feature_columns,
        epochs=finetune_epochs,
        lr=0.005,  # lower LR for fine-tuning
        seed=seed + 1,
        encoder=community_result["encoder"],
        device=device,
    )

    return {
        "shared_environments": len(common_envs),
        "community_pretrain": {
            "final_loss": community_result["final_loss"],
            "collapse_diagnostics": community_result["collapse_diagnostics"],
        },
        "weather_finetune": {
            "final_loss": finetune_result["final_loss"],
            "collapse_diagnostics": finetune_result["collapse_diagnostics"],
        },
        "status": "completed",
    }

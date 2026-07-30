"""Genotype block encoder (TDD Section 6.1, Gate D).

Wraps ``TokenSequenceEncoder`` for genotype LD-block sequences — each genotype
is a sequence of blocks ordered by chromosome/position, and the encoder learns
block-level representations that can be pretrained via masked reconstruction
and then used in the G×E model.

Architecture follows the same pattern as ``SharedEnvironmentEncoder``
(environment_encoder.py), per TDD 6.1's "both roles use the same
TokenSequenceEncoder class below" — only the input features differ.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from plant_context.models.context_encoder import TokenSequenceEncoder


class GenotypeBlockEncoder(nn.Module):
    """TokenSequenceEncoder wrapper for genotype LD-block sequences.

    Each genotype's blocks are ordered by chromosome/position. Missing blocks
    (blocks a genotype does not have markers for) are handled via a padding
    mask.

    Parameters
    ----------
    n_block_features :
        Number of features per block (e.g. 1 for mean_dosage only, 2 if
        including n_markers).
    d_model, n_heads, n_layers :
        Transformer architecture — kept small per TDD 6.1.
    """

    def __init__(
        self,
        n_block_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = TokenSequenceEncoder(
            n_features=n_block_features, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )
        self._n_block_features = n_block_features
        self._d_model = d_model

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_block_features(self) -> int:
        return self._n_block_features

    def forward(
        self, block_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of genotype block sequences.

        Parameters
        ----------
        block_tokens :
            ``[batch, n_blocks, n_block_features]`` tensor, ordered by
            chromosome/position.
        padding_mask :
            ``[batch, n_blocks]`` bool, True = this block is absent/should be
            masked from attention.  If None, all blocks assumed present.

        Returns
        -------
        block_embeddings : ``[batch, n_blocks, d_model]``
        pooled : ``[batch, d_model]``
        """
        return self.encoder(block_tokens, key_padding_mask=padding_mask)

    @torch.no_grad()
    def embed_genotypes(
        self,
        per_genotype_tokens: dict,
        feature_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Convert per-genotype block token dict into pooled embeddings.

        Parameters
        ----------
        per_genotype_tokens :
            ``{genotype_id: DataFrame indexed by ld_block_id,
            columns = feature_columns}``.
        feature_columns :
            Which columns to use as features.

        Returns
        -------
        DataFrame indexed by genotype_id with ``d_model`` embedding columns
        named ``geno_embed_0`` … ``geno_embed_{d_model-1}``.
        """
        data, mask, _, sample_ids = _build_block_tensor(
            per_genotype_tokens, feature_columns
        )
        self.eval()
        _, pooled = self.forward(
            torch.tensor(data),
            padding_mask=torch.tensor(mask),
        )
        return pd.DataFrame(
            pooled.numpy(),
            index=sample_ids,
            columns=[f"geno_embed_{i}" for i in range(self._d_model)],
        )


def _build_block_tensor(
    per_genotype_tokens: dict,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Build padded ``[n_genotypes, max_blocks, n_features]`` tensor + mask.

    Returns (data, padding_mask, token_ids_per_genotype, genotype_ids).
    """
    sample_ids = list(per_genotype_tokens.keys())
    max_len = max(len(df) for df in per_genotype_tokens.values()) if sample_ids else 0
    n_features = len(feature_columns)

    data = np.zeros((len(sample_ids), max_len, n_features), dtype=np.float32)
    mask = np.ones((len(sample_ids), max_len), dtype=bool)
    token_ids_per_genotype = []

    for i, gid in enumerate(sample_ids):
        df = per_genotype_tokens[gid]
        token_ids = list(df.index)
        token_ids_per_genotype.append(token_ids)
        n = len(df)
        data[i, :n, :] = df[feature_columns].to_numpy(dtype=np.float32)
        mask[i, :n] = False

    return data, mask, token_ids_per_genotype, sample_ids


def _ordered_per_sample_dict(
    long_tokens: pd.DataFrame,
    id_col: str,
    order_col: str,
    canonical_order: list,
    feature_columns: Sequence[str],
) -> dict:
    """Convert long-format tokens to per-sample dict with deterministic order.

    ``long_tokens`` (one row per ``id_col`` × ``order_col``) →
    ``{sample_id: DataFrame indexed by order_col}``, rows reindexed to
    ``canonical_order`` subsequence actually present for that sample.
    """
    per_sample = {}
    for sample_id, group in long_tokens.groupby(id_col):
        indexed = group.set_index(order_col)
        present_in_order = [c for c in canonical_order if c in indexed.index]
        per_sample[sample_id] = indexed.loc[present_in_order, feature_columns]
    return per_sample


def pretrain_genotype_encoder(
    genotype_tokens_long: pd.DataFrame,
    block_order: list,
    feature_columns: Sequence[str] = ("mean_dosage",),
    mask_fraction: float = 0.15,
    d_model: int = 32,
    epochs: int = 80,
    lr: float = 0.01,
    seed: int = 1234,
    device: str = "cpu",
) -> dict:
    """Pretrain a ``GenotypeBlockEncoder`` via masked reconstruction.

    Parameters
    ----------
    genotype_tokens_long :
        Output of ``tokenize_genotype_blocks`` (genotype × block tokens,
        long format).
    block_order :
        Ordered list of ``ld_block_id`` values (deterministic chrom/pos order).
    feature_columns :
        Which token columns to use as features.
    mask_fraction :
        Fraction of blocks to mask per genotype (contiguous run).
    d_model, epochs, lr, seed :
        Training hyperparameters.

    Returns a dict with ``encoder`` (GenotypeBlockEncoder), ``head``,
    ``loss_history``, ``final_loss``, and ``collapse_diagnostics``.
    """
    import functools

    from plant_context.models.pretraining import pretrain_masked_reconstruction
    from plant_context.tokenizers.masking import mask_contiguous_run

    per_sample = _ordered_per_sample_dict(
        genotype_tokens_long, "genotype_id", "ld_block_id",
        block_order, feature_columns,
    )
    # Drop genotypes with fewer than 2 blocks (nothing to mask-reconstruct)
    per_sample = {k: v for k, v in per_sample.items() if len(v) >= 2}

    mask_fn = functools.partial(mask_contiguous_run, mask_fraction=mask_fraction)

    result = pretrain_masked_reconstruction(
        per_sample_tokens=per_sample,
        feature_columns=list(feature_columns),
        mask_fn=mask_fn,
        d_model=d_model,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
    )

    # Wrap the generic TokenSequenceEncoder in a GenotypeBlockEncoder
    n_features = len(feature_columns)
    enc = GenotypeBlockEncoder(n_block_features=n_features, d_model=d_model)
    enc.encoder.load_state_dict(result["encoder"].state_dict())
    result["encoder"] = enc

    return result

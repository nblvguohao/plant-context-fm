"""Shared token-sequence encoder (TDD Section 6.1, 15 item 9).

One small Transformer encoder class, instantiated separately per domain: an
environment-stage-encoder instance is the piece TDD 6.1 actually calls
"shared" -- meant to transfer between the community and G-E branches --
while a genotype-block-encoder instance stays domain-specific (TDD Section
2: "the community encoder and genotype encoder stay domain-specific; only
the environment representation is meant to be shared"). Both roles use the
same ``TokenSequenceEncoder`` class below -- there is no architectural
reason for them to differ, only the weights and the domain of tokens fed in
do.

Candidate architectures were ranked in TDD 6.1: (1) a small Transformer/TCN,
(2) stage-aware attention, (3) a state-space model, "only once proven
necessary for long sequences". Our sequences (5-6 environment stages, tens
of genotype blocks) are short, so (1) -- a small Transformer -- is used;
(2)/(3) are not implemented.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TokenSequenceEncoder(nn.Module):
    """A small Transformer over a sequence of feature-vector tokens.

    Input: ``tokens`` [batch, n_tokens, n_features] plus an optional
    ``key_padding_mask`` [batch, n_tokens] (True = ignore this position,
    e.g. a token this sample does not have -- TDD 6.1 requires supporting a
    missing mask). Output: per-token embeddings [batch, n_tokens, d_model]
    and a pooled embedding [batch, d_model] (mean over non-masked tokens).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ):
        x = self.input_proj(tokens)
        if key_padding_mask is not None:
            # Zero out masked positions BEFORE the transformer, avoiding the
            # nested-tensor prototype implementation that can produce NaN.
            x = x * (~key_padding_mask).unsqueeze(-1).float()
        encoded = self.transformer(x)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)
        return encoded, pooled


def embedding_collapse_diagnostics(pooled_embeddings: torch.Tensor) -> dict:
    """Per-dimension std, effective rank, of a batch of pooled embeddings.

    TDD Section 7.1: "monitor per-dimension std, effective rank, and
    covariance; set an automatic collapse-failure gate" -- this computes
    the diagnostics, ``check_collapse_gate`` below applies the gate.

    Effective rank (Roy & Vetterli 2007): exp(entropy of the normalized
    singular-value spectrum of the centered embeddings) -- 1.0 for a
    rank-1 (fully collapsed) embedding, up to min(batch, d_model) for an
    isotropic one.
    """
    x = pooled_embeddings.detach()
    if x.shape[0] < 2:
        raise ValueError("need at least 2 samples to assess collapse")
    centered = x - x.mean(dim=0, keepdim=True)
    per_dim_std = centered.std(dim=0)

    singular_values = torch.linalg.svdvals(centered)
    sv = singular_values[singular_values > 1e-12]
    if len(sv) == 0:
        effective_rank = 0.0
    else:
        p = sv / sv.sum()
        entropy = -(p * p.log()).sum()
        effective_rank = float(torch.exp(entropy))

    return {
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_min": float(per_dim_std.min()),
        "effective_rank": effective_rank,
    }


def check_collapse_gate(
    diagnostics: dict, min_effective_rank: float = 2.0, min_per_dim_std: float = 1e-3
) -> list:
    """Return a list of violated-gate messages; an empty list means the gate passes."""
    violations = []
    if diagnostics["effective_rank"] < min_effective_rank:
        violations.append(
            f"effective_rank {diagnostics['effective_rank']:.3f} < {min_effective_rank} -- "
            "embeddings look collapsed toward a low-dimensional (or single) point"
        )
    if diagnostics["per_dim_std_min"] < min_per_dim_std:
        violations.append(
            f"per_dim_std_min {diagnostics['per_dim_std_min']:.6f} < {min_per_dim_std} -- "
            "at least one embedding dimension is nearly constant across the batch"
        )
    return violations

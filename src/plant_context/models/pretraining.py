"""Masked-token-reconstruction self-supervised pretraining (TDD Section 7,
15 item 9).

Generic across genotype-block and environment-stage tokens: given a
per-sample ordered sequence of tokens (each a feature vector) and a masking
function (tokenizers/masking.py) that picks which tokens to hide, trains a
TokenSequenceEncoder plus a small linear head to reconstruct the *original*
(pre-mask) feature vector at the masked positions only (MSE loss). This is
the "predict the raw value directly" style of pretraining objective (TDD
7.1's first bullet: predict the masked block's allele-dosage distribution
-- simplified here to point-estimate regression of mean_dosage, not a full
distribution). Because the target is the actual observed feature value, not
a learned/EMA-teacher embedding, there is no representation-collapse route
through the *target* side; the EMA-teacher machinery TDD 7.1 requires for
"predicting hidden representations" is deliberately not built here, since
this simpler objective does not need it. The pooled embedding *output* of
the encoder can still collapse for other reasons (e.g. capacity that never
learns anything useful), which is exactly what
embedding_collapse_diagnostics/check_collapse_gate (context_encoder.py)
watch for -- this module surfaces both on every run rather than only on
request.
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

MaskFn = Callable[[Sequence, int], set]


def _pad_sample_sequences(per_sample_tokens: dict, feature_columns: list):
    """dict[sample_id -> DataFrame indexed by token_id, columns =
    feature_columns] -> a padded [n_samples, max_len, n_features] array, a
    boolean padding mask [n_samples, max_len] (True = padding), and the
    token-id sequence per sample (in whatever order the input DataFrame's
    index already has -- not re-sorted).
    """
    sample_ids = list(per_sample_tokens.keys())
    max_len = max(len(df) for df in per_sample_tokens.values())
    n_features = len(feature_columns)

    features = np.zeros((len(sample_ids), max_len, n_features), dtype=np.float32)
    padding_mask = np.ones((len(sample_ids), max_len), dtype=bool)
    token_ids_per_sample = []

    for i, sid in enumerate(sample_ids):
        df = per_sample_tokens[sid]
        token_ids = list(df.index)
        token_ids_per_sample.append(token_ids)
        n = len(df)
        features[i, :n, :] = df[feature_columns].to_numpy(dtype=np.float32)
        padding_mask[i, :n] = False

    return features, padding_mask, token_ids_per_sample, sample_ids


def pretrain_masked_reconstruction(
    per_sample_tokens: dict,
    feature_columns: list,
    mask_fn: MaskFn,
    d_model: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 1234,
    device: str = "cpu",
    encoder: Optional[nn.Module] = None,
) -> dict:
    """Train a TokenSequenceEncoder with a masked-reconstruction objective.

    ``per_sample_tokens``: ``{sample_id: DataFrame indexed by token_id,
    columns = feature_columns}``, already in the order that makes
    contiguous masking meaningful for whichever ``mask_fn`` is used.
    ``mask_fn(token_ids, seed=...) -> set`` is called once per sample with
    that sample's ordered token-id list and a per-sample-varying seed
    (passed as the keyword ``seed``, since callers typically bind
    mask_fraction/group_of/frequency_of via ``functools.partial`` with a
    keyword too, and a positional seed would collide whenever it lands on
    the same argument slot), and must return the set of token ids to hide
    from the encoder input and use as the reconstruction target.

    ``encoder`` may be a pre-initialised ``TokenSequenceEncoder`` (or any
    module with the same forward signature) for fine-tuning from a
    pretrained checkpoint. If None, a fresh encoder is created.

    Returns a dict with the trained encoder, the reconstruction head, the
    loss history, and collapse diagnostics computed on this run's final
    pooled embeddings.
    """
    torch.manual_seed(seed)
    features, padding_mask, token_ids_per_sample, sample_ids = _pad_sample_sequences(
        per_sample_tokens, feature_columns
    )
    n_samples, max_len, n_features = features.shape

    mask_positions = np.zeros((n_samples, max_len), dtype=bool)
    for i, token_ids in enumerate(token_ids_per_sample):
        # `seed` passed as a keyword: a caller typically binds mask_fraction
        # (or group_of/frequency_of) via functools.partial with a keyword,
        # and calling positionally here would collide with that binding
        # whenever it lands on the same argument slot.
        masked_ids = mask_fn(token_ids, seed=seed + i)
        for pos, tid in enumerate(token_ids):
            if tid in masked_ids:
                mask_positions[i, pos] = True

    non_padding = ~padding_mask
    feature_mean = features[non_padding].mean(axis=0)
    feature_std = features[non_padding].std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    features_scaled = (features - feature_mean) / feature_std

    encoder_input = features_scaled.copy()
    # Masked tokens get a neutral placeholder (0.0 is the post-standardization
    # mean), not the true value they are being asked to reconstruct.
    encoder_input[mask_positions] = 0.0

    if encoder is None:
        encoder = TokenSequenceEncoder(
            n_features=n_features, d_model=d_model, n_heads=n_heads, n_layers=n_layers
        ).to(device)
    else:
        encoder = encoder.to(device)
    head = nn.Linear(encoder.d_model if hasattr(encoder, "d_model") else d_model, n_features).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    encoder_input_t = torch.tensor(encoder_input, dtype=torch.float32, device=device)
    padding_mask_t = torch.tensor(padding_mask, device=device)
    targets_t = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    mask_positions_t = torch.tensor(mask_positions, device=device)

    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        encoded, _pooled = encoder(encoder_input_t, key_padding_mask=padding_mask_t)
        predicted = head(encoded)
        if mask_positions_t.any():
            loss = ((predicted - targets_t)[mask_positions_t] ** 2).mean()
        else:
            loss = torch.tensor(0.0)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    with torch.no_grad():
        _, final_pooled = encoder(encoder_input_t, key_padding_mask=padding_mask_t)
    diagnostics = embedding_collapse_diagnostics(final_pooled) if n_samples >= 2 else {}
    collapse_violations = check_collapse_gate(diagnostics) if diagnostics else []

    return {
        "encoder": encoder,
        "head": head,
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "collapse_diagnostics": diagnostics,
        "collapse_violations": collapse_violations,
        "sample_ids": sample_ids,
    }

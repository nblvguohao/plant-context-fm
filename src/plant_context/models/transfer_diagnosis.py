"""Cross-domain transfer diagnosis for SharedEnvironmentEncoder.

TDD Section 6.1 requires a failure-diagnosis protocol when the environment
encoder pretrained on community data is transferred to G×E (weather) data.
The protocol must distinguish:

- domain-gap failure (distributions are too different for transfer to help);
- capacity/optimization failure (distributions overlap but the encoder cannot
  learn the target task without more capacity or tuning);
- successful transfer.

This module provides reusable diagnostics: domain-difference metrics,
baseline comparisons, layer-wise fine-tuning ablation, and a failure-mode
classifier.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from plant_context.models.environment_encoder import SharedEnvironmentEncoder


def _build_tensors_for_features(
    encoder: SharedEnvironmentEncoder,
    features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reusable tensor builder that works with either weather or community features."""
    return encoder._build_tensor(features, stage_order, feature_columns)


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Maximum Mean Discrepancy (MMD) with RBF kernel between two feature arrays.

    A large MMD indicates that the two domains occupy different regions of
    feature space. This is used to quantify the community/weather domain gap.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Drop rows with any NaN
    x = x[~np.isnan(x).any(axis=1)]
    y = y[~np.isnan(y).any(axis=1)]

    if len(x) == 0 or len(y) == 0:
        return float("nan")

    if gamma is None:
        # Median heuristic for the bandwidth
        pairwise = np.concatenate([x, y], axis=0)
        dists = np.linalg.norm(pairwise[:, None, :] - pairwise[None, :, :], axis=2)
        gamma = 1.0 / (2.0 * (np.median(dists[dists > 0]) ** 2))

    def rbf_kernel(a, b):
        return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))

    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)

    # Unbiased MMD^2 estimator
    mmd2 = (
        k_xx.sum() - np.trace(k_xx)
    ) / (len(x) * (len(x) - 1)) + (
        k_yy.sum() - np.trace(k_yy)
    ) / (len(y) * (len(y) - 1)) - 2 * k_xy.mean()

    return float(np.sqrt(max(mmd2, 0.0)))


def compute_wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    """Mean per-feature 1-Wasserstein distance between source and target arrays.

    Uses the closed-form CDF difference for 1-D distributions. We average
    across features to get a single interpretable number.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same number of features")

    distances = []
    for col in range(source.shape[1]):
        s = source[:, col]
        t = target[:, col]
        s = s[~np.isnan(s)]
        t = t[~np.isnan(t)]
        if len(s) == 0 or len(t) == 0:
            distances.append(float("nan"))
            continue
        # 1-Wasserstein between two empirical distributions
        sorted_s = np.sort(s)
        sorted_t = np.sort(t)
        # Equal-weight CDFs: interpolate to common quantile grid
        n = max(len(sorted_s), len(sorted_t))
        qs = np.linspace(0, 1, n)
        q_s = np.quantile(sorted_s, qs)
        q_t = np.quantile(sorted_t, qs)
        distances.append(float(np.mean(np.abs(q_s - q_t))))

    return float(np.nanmean(distances))


def domain_difference_report(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
) -> dict:
    """Compute interpretable domain-difference metrics.

    Returns MMD, mean Wasserstein distance, and per-feature Wasserstein
    distances. The per-feature distances help diagnose *which* environmental
    variables drive the gap.
    """
    common_cols = source_features.columns.intersection(target_features.columns)
    if len(common_cols) == 0:
        return {
            "mmd": float("nan"),
            "mean_wasserstein": float("nan"),
            "per_feature_wasserstein": {},
            "n_source": len(source_features),
            "n_target": len(target_features),
            "n_common_features": 0,
        }

    s = source_features[common_cols].to_numpy(dtype=np.float64)
    t = target_features[common_cols].to_numpy(dtype=np.float64)

    per_feature = {}
    for i, col in enumerate(common_cols):
        sc = s[:, i]
        tc = t[:, i]
        sc = sc[~np.isnan(sc)]
        tc = tc[~np.isnan(tc)]
        if len(sc) == 0 or len(tc) == 0:
            per_feature[col] = float("nan")
        else:
            qs = np.linspace(0, 1, max(len(sc), len(tc)))
            per_feature[col] = float(
                np.mean(np.abs(np.quantile(np.sort(sc), qs) - np.quantile(np.sort(tc), qs)))
            )

    return {
        "mmd": compute_mmd(s, t),
        "mean_wasserstein": compute_wasserstein_distance(s, t),
        "per_feature_wasserstein": per_feature,
        "n_source": len(source_features),
        "n_target": len(target_features),
        "n_common_features": len(common_cols),
    }


def _masked_reconstruction_loss(
    encoder: SharedEnvironmentEncoder,
    features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
    mask_fraction: float = 0.2,
    seed: int = 0,
) -> float:
    """A simple downstream-task proxy: masked stage reconstruction loss.

    This is comparable across frozen/random/in-domain initializations and
    serves as the metric for the baseline comparison and layer ablation.
    """
    tensor, mask = _build_tensors_for_features(encoder, features, stage_order, feature_columns)

    rng = np.random.default_rng(seed)
    n_envs, n_stages, n_features = tensor.shape
    n_mask = max(1, int(mask_fraction * n_stages))

    # Mask a contiguous run of stages per environment (simple deterministic-ish mask)
    masked_tensor = tensor.clone()
    target = tensor.clone()
    eval_mask = torch.zeros_like(mask, dtype=torch.bool)
    for i in range(n_envs):
        valid_stages = (~mask[i]).nonzero(as_tuple=True)[0].tolist()
        if len(valid_stages) < n_mask + 1:
            continue
        start = rng.integers(0, len(valid_stages) - n_mask)
        for j in valid_stages[start : start + n_mask]:
            masked_tensor[i, j, :] = 0.0
            eval_mask[i, j] = True

    encoder.eval()
    with torch.no_grad():
        encoded, _ = encoder(masked_tensor, stage_mask=mask)
        # Reconstruct original features with a linear head fitted on-the-fly
        # (a fixed random head would be too noisy; we fit a ridge regression head)
        train_idx = eval_mask.any(dim=1)  # environments with at least one masked stage
        if train_idx.sum() == 0:
            return float("nan")

        flat_encoded = encoded[eval_mask].numpy()
        flat_target = target[eval_mask].numpy()

        # Ridge regression closed form
        lamb = 1e-3
        xtx = flat_encoded.T @ flat_encoded + lamb * np.eye(flat_encoded.shape[1])
        xty = flat_encoded.T @ flat_target
        try:
            head = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            return float("nan")

        pred = flat_encoded @ head
        mse = float(np.mean((pred - flat_target) ** 2))

    return mse


def baseline_transfer_comparison(
    source_features: pd.DataFrame,
    target_features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
    pretrain_fn: Callable,
    pretrain_kwargs: dict,
    evaluation_seed: int = 42,
) -> dict:
    """Compare three initializations on the target task.

    - frozen: encoder pretrained on source, output head trained on target
    - random_init: same architecture, randomly initialized, trained on target
    - in_domain: encoder trained from scratch on target (upper bound)

    ``pretrain_fn`` should accept ``features, stage_order, feature_columns``
    plus ``**pretrain_kwargs`` and return a dict with an ``"encoder"`` key
    containing a ``SharedEnvironmentEncoder``.
    """
    common_envs = source_features.index.intersection(target_features.index)
    if len(common_envs) == 0:
        return {
            "status": "no_shared_environments",
            "frozen_loss": None,
            "random_init_loss": None,
            "in_domain_loss": None,
        }

    target_sub = target_features.loc[common_envs]

    # In-domain baseline: train on target from scratch
    in_domain_result = pretrain_fn(
        target_sub,
        stage_order=stage_order,
        feature_columns=feature_columns,
        **pretrain_kwargs,
    )
    in_domain_encoder = in_domain_result["encoder"]
    in_domain_loss = _masked_reconstruction_loss(
        in_domain_encoder, target_sub, stage_order, feature_columns, seed=evaluation_seed
    )

    # Random-init baseline: fresh encoder, same architecture
    random_encoder = SharedEnvironmentEncoder(
        n_stage_features=len(feature_columns),
        d_model=in_domain_encoder.d_model,
        stage_names=stage_order,
    )
    random_loss = _masked_reconstruction_loss(
        random_encoder, target_sub, stage_order, feature_columns, seed=evaluation_seed + 1
    )

    # Frozen source-pretrained baseline
    source_sub = source_features.loc[common_envs]
    source_result = pretrain_fn(
        source_sub,
        stage_order=stage_order,
        feature_columns=feature_columns,
        **pretrain_kwargs,
    )
    frozen_encoder = source_result["encoder"]
    frozen_loss = _masked_reconstruction_loss(
        frozen_encoder, target_sub, stage_order, feature_columns, seed=evaluation_seed + 2
    )

    return {
        "status": "completed",
        "n_shared_environments": len(common_envs),
        "frozen_loss": frozen_loss,
        "random_init_loss": random_loss,
        "in_domain_loss": in_domain_loss,
    }


def layer_wise_finetune_ablation(
    source_pretrained_encoder: SharedEnvironmentEncoder,
    target_features: pd.DataFrame,
    stage_order: Sequence[str],
    feature_columns: Sequence[str],
    finetune_fn: Callable,
    finetune_kwargs: dict,
    evaluation_seed: int = 100,
) -> dict:
    """Test how many layers need to be unfrozen to recover target performance.

    Freezing strategies tried:
      - all: only the output reconstruction head is trained
      - head: input projection + final transformer layer trainable
      - progressive: unfreeze last k layers
      - full: all parameters trainable

    Returns losses for each strategy.
    """
    results = {}
    base_state = source_pretrained_encoder.state_dict()

    # Discover the last transformer layer name dynamically so this works
    # regardless of n_layers.
    layer_names = [n for n, _ in source_pretrained_encoder.named_parameters() if n.startswith("encoder.transformer.layers.")]
    last_layer_prefix = None
    if layer_names:
        # layer names look like encoder.transformer.layers.0.self_attn...
        layer_indices = sorted(set(int(n.split(".")[3]) for n in layer_names))
        last_layer_prefix = f"encoder.transformer.layers.{layer_indices[-1]}"

    strategies = {
        "frozen_all": [],
        "unfreeze_head": ["input_proj"],
        "unfreeze_last_layer": ["input_proj", last_layer_prefix] if last_layer_prefix else ["input_proj"],
        "unfreeze_all": None,  # None means no parameter freezing
    }

    for name, allowed in strategies.items():
        encoder = SharedEnvironmentEncoder(
            n_stage_features=len(feature_columns),
            d_model=source_pretrained_encoder.d_model,
            n_layers=source_pretrained_encoder.n_layers,
            n_heads=source_pretrained_encoder.n_heads,
            stage_names=stage_order,
        )
        encoder.load_state_dict(base_state)

        if allowed is not None:
            for param_name, param in encoder.named_parameters():
                param.requires_grad = any(param_name.startswith(a) for a in allowed)
        else:
            for param in encoder.parameters():
                param.requires_grad = True

        # Fine-tune (caller-supplied function is responsible for respecting
        # requires_grad because we pass the encoder in).
        finetuned = finetune_fn(
            encoder,
            target_features,
            stage_order=stage_order,
            feature_columns=feature_columns,
            **finetune_kwargs,
        )
        loss = _masked_reconstruction_loss(
            finetuned, target_features, stage_order, feature_columns, seed=evaluation_seed
        )
        results[name] = loss

    return results


def classify_transfer_failure(
    baseline_results: dict,
    domain_report: dict,
    layer_ablation: dict,
    mmd_threshold: float = 0.5,
    wasserstein_threshold: float = 1.0,
    improvement_threshold: float = 0.05,
) -> dict:
    """Classify the transfer outcome per TDD 6.1 failure-diagnosis protocol.

    Rules:
      - domain_gap: frozen encoder not better than random AND domain metrics high
      - capacity_failure: domain metrics low but frozen not better than random,
        and full fine-tuning recovers in-domain performance
      - successful_transfer: frozen encoder matches or beats random init
      - inconclusive: none of the above
    """
    frozen = baseline_results.get("frozen_loss")
    random = baseline_results.get("random_init_loss")
    in_domain = baseline_results.get("in_domain_loss")

    mmd = domain_report.get("mmd", float("nan"))
    wasserstein = domain_report.get("mean_wasserstein", float("nan"))
    full_finetune = layer_ablation.get("unfreeze_all")

    domain_gap_high = (
        not np.isnan(mmd) and mmd > mmd_threshold
    ) or (
        not np.isnan(wasserstein) and wasserstein > wasserstein_threshold
    )

    def rel_improvement(better, worse):
        if better is None or worse is None or np.isnan(better) or np.isnan(worse):
            return 0.0
        if worse == 0:
            return 0.0
        return (worse - better) / abs(worse)

    frozen_better_than_random = rel_improvement(frozen, random) > improvement_threshold
    # "Recovers" means full fine-tune is close to in-domain (within 25% relative)
    full_recovers = (
        full_finetune is not None
        and not np.isnan(full_finetune)
        and abs(rel_improvement(in_domain, full_finetune)) < 0.25
    )
    # Fine-tuning must actually help over the frozen encoder to call it a
    # capacity/optimization failure rather than just a hard task.
    finetune_helps = rel_improvement(full_finetune, frozen) > improvement_threshold

    if frozen_better_than_random:
        mode = "successful_transfer"
        reason = "frozen source encoder outperforms random initialization"
    elif domain_gap_high and not full_recovers:
        mode = "domain_gap_failure"
        reason = (
            f"domain metrics high (MMD={mmd:.3f}, Wasserstein={wasserstein:.3f}) "
            "and full fine-tuning does not recover in-domain performance"
        )
    elif not domain_gap_high and full_recovers and finetune_helps:
        mode = "capacity_failure"
        reason = (
            "domain distributions overlap but frozen encoder underperforms; "
            "full fine-tuning recovers in-domain performance"
        )
    else:
        mode = "inconclusive"
        reason = "pattern does not clearly match domain-gap or capacity failure"

    return {
        "failure_mode": mode,
        "reason": reason,
        "domain_gap_high": domain_gap_high,
        "frozen_better_than_random": frozen_better_than_random,
        "full_finetune_recovers": full_recovers,
    }

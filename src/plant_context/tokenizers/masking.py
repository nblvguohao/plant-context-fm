"""Structured masking strategies for self-supervised pretraining (TDD
Section 7, 15 item 9).

Each function decides WHICH tokens in one already-ordered token sequence
get masked; none of them touch model weights or losses (see
models/pretraining.py for the training loop that uses these). "Ordered"
matters: contiguous-run masking only makes sense given a meaningful order
(chromosome/position for genotype blocks, chronological growth stage for
environment stages) -- TDD Section 5.2 explicitly forbids treating an
arbitrary statistic-sorted order as if it were this kind of biological/
temporal sequence, so callers must pass tokens already correctly ordered by
the corresponding tokenizer (tokenizers/genotype.py, tokenizers/
environment.py), never re-sorted here.

Every function's last positional argument is ``seed`` (even the ones that
don't strictly need randomness for a trivial case), so all four have the
same ``(token_ids, ..., seed) -> set`` shape and can be used interchangeably
as a ``mask_fn`` by models/pretraining.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mask_random_tokens(token_ids: Sequence, mask_fraction: float, seed: int) -> set:
    """Uniformly random subset of ``token_ids``, ~mask_fraction of them."""
    token_ids = list(token_ids)
    n = len(token_ids)
    if n == 0:
        return set()
    n_mask = max(1, min(n, round(n * mask_fraction)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=n_mask, replace=False)
    return {token_ids[i] for i in chosen}


def mask_contiguous_run(ordered_token_ids: Sequence, mask_fraction: float, seed: int) -> set:
    """A single contiguous run of ~mask_fraction of ``ordered_token_ids``.

    ``ordered_token_ids`` must already be in the order that makes
    "contiguous" meaningful -- this function does not sort or validate that
    ordering, it only walks it.
    """
    ordered_token_ids = list(ordered_token_ids)
    n = len(ordered_token_ids)
    if n == 0:
        return set()
    run_length = max(1, min(n, round(n * mask_fraction)))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n - run_length + 1))
    return set(ordered_token_ids[start : start + run_length])


def mask_whole_groups(
    token_ids: Sequence, group_of: dict, mask_fraction_of_groups: float, seed: int
) -> set:
    """Mask every token belonging to a randomly-chosen subset of groups.

    ``group_of`` maps each token id to its group id (e.g. genus for a
    species token, chromosome for a marker). ~mask_fraction_of_groups of the
    *groups* present are chosen, and every token in a chosen group is
    masked -- not a per-token fraction, since the point is to remove entire
    related clusters at once (TDD 5.1's "whole genus/family mask").
    """
    token_ids = list(token_ids)
    groups = sorted({group_of[t] for t in token_ids if t in group_of})
    if not groups:
        return set()
    n_mask_groups = max(1, min(len(groups), round(len(groups) * mask_fraction_of_groups)))
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(groups), size=n_mask_groups, replace=False)
    chosen_groups = {groups[i] for i in chosen_idx}
    return {t for t in token_ids if group_of.get(t) in chosen_groups}


def mask_rare_weighted_tokens(
    token_ids: Sequence, frequency_of: dict, mask_fraction: float, seed: int
) -> set:
    """Random subset of ``token_ids``, oversampling rarer tokens (TDD 5.1's
    "rare-species-enhanced mask"): sampling weight is inverse frequency, so
    a token seen once across the whole corpus is far more likely to be
    selected than a token seen thousands of times.
    """
    token_ids = list(token_ids)
    n = len(token_ids)
    if n == 0:
        return set()
    n_mask = max(1, min(n, round(n * mask_fraction)))
    freqs = np.array([max(frequency_of.get(t, 1), 1) for t in token_ids], dtype=float)
    weights = 1.0 / freqs
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=n_mask, replace=False, p=weights)
    return {token_ids[i] for i in chosen}

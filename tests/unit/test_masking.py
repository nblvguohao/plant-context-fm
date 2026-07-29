import numpy as np
import pytest

from plant_context.tokenizers.masking import (
    mask_contiguous_run,
    mask_random_tokens,
    mask_rare_weighted_tokens,
    mask_whole_groups,
)


def test_mask_random_tokens_masks_expected_fraction():
    tokens = list(range(20))
    masked = mask_random_tokens(tokens, mask_fraction=0.3, seed=1)
    assert len(masked) == 6


def test_mask_random_tokens_deterministic_given_seed():
    tokens = list(range(20))
    a = mask_random_tokens(tokens, mask_fraction=0.3, seed=42)
    b = mask_random_tokens(tokens, mask_fraction=0.3, seed=42)
    assert a == b


def test_mask_random_tokens_differs_across_seeds():
    tokens = list(range(20))
    a = mask_random_tokens(tokens, mask_fraction=0.3, seed=1)
    b = mask_random_tokens(tokens, mask_fraction=0.3, seed=2)
    assert a != b


def test_mask_random_tokens_empty_input():
    assert mask_random_tokens([], mask_fraction=0.3, seed=1) == set()


def test_mask_contiguous_run_is_actually_contiguous_in_given_order():
    tokens = [f"b{i}" for i in range(20)]
    masked = mask_contiguous_run(tokens, mask_fraction=0.25, seed=7)
    positions = sorted(tokens.index(t) for t in masked)
    assert positions == list(range(positions[0], positions[0] + len(positions)))
    assert len(masked) == 5


def test_mask_contiguous_run_never_wraps_past_the_end():
    tokens = list(range(10))
    for seed in range(50):
        masked = mask_contiguous_run(tokens, mask_fraction=0.4, seed=seed)
        positions = sorted(tokens.index(t) for t in masked)
        assert positions[-1] < len(tokens)
        assert positions == list(range(positions[0], positions[-1] + 1))


def test_mask_contiguous_run_deterministic_given_seed():
    tokens = list(range(20))
    a = mask_contiguous_run(tokens, mask_fraction=0.3, seed=42)
    b = mask_contiguous_run(tokens, mask_fraction=0.3, seed=42)
    assert a == b


def test_mask_whole_groups_never_partially_masks_a_group():
    tokens = ["s1", "s2", "s3", "s4", "s5", "s6"]
    group_of = {"s1": "genusA", "s2": "genusA", "s3": "genusB", "s4": "genusB", "s5": "genusC", "s6": "genusC"}
    masked = mask_whole_groups(tokens, group_of, mask_fraction_of_groups=0.34, seed=3)

    masked_groups = {group_of[t] for t in masked}
    for g in masked_groups:
        group_members = {t for t in tokens if group_of[t] == g}
        assert group_members.issubset(masked), f"group {g} was only partially masked"

    unmasked = set(tokens) - masked
    unmasked_groups = {group_of[t] for t in unmasked}
    assert masked_groups.isdisjoint(unmasked_groups)


def test_mask_whole_groups_handles_tokens_with_no_group():
    tokens = ["s1", "s2"]
    assert mask_whole_groups(tokens, {}, mask_fraction_of_groups=0.5, seed=1) == set()


def test_mask_rare_weighted_tokens_prefers_rare_over_common():
    tokens = ["common", "rare"]
    frequency_of = {"common": 1000, "rare": 1}
    rare_picks = 0
    n_trials = 200
    for seed in range(n_trials):
        masked = mask_rare_weighted_tokens(tokens, frequency_of, mask_fraction=0.5, seed=seed)
        if "rare" in masked:
            rare_picks += 1
    assert rare_picks / n_trials > 0.9


def test_mask_rare_weighted_tokens_masks_expected_count():
    tokens = list(range(10))
    frequency_of = {t: 1 for t in tokens}
    masked = mask_rare_weighted_tokens(tokens, frequency_of, mask_fraction=0.3, seed=1)
    assert len(masked) == 3

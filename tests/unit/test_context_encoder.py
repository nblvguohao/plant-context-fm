import torch
import pytest

from plant_context.models.context_encoder import (
    TokenSequenceEncoder,
    check_collapse_gate,
    embedding_collapse_diagnostics,
)


def test_token_sequence_encoder_forward_shape():
    encoder = TokenSequenceEncoder(n_features=6, d_model=16, n_heads=2, n_layers=1)
    tokens = torch.randn(5, 8, 6)
    encoded, pooled = encoder(tokens)
    assert encoded.shape == (5, 8, 16)
    assert pooled.shape == (5, 16)


def test_token_sequence_encoder_padded_content_has_zero_effect_on_valid_positions():
    # Two versions of the same batch, differing ONLY in the value placed at
    # a position marked as padding. If padding is handled correctly (both
    # by the transformer's own attention masking and by this module's
    # pooling), the valid positions' encoded output and the pooled output
    # must be identical between the two versions -- not just "our own
    # averaging formula happens to ignore that slot", but "the padded
    # content never influenced anything through attention either".
    torch.manual_seed(0)
    encoder = TokenSequenceEncoder(n_features=4, d_model=8, n_heads=2, n_layers=1)
    encoder.eval()

    tokens_a = torch.randn(2, 3, 4)
    tokens_a[1, 2, :] = 0.0
    tokens_b = tokens_a.clone()
    tokens_b[1, 2, :] = 999.0  # only the padded slot's content differs

    padding_mask = torch.tensor([[False, False, False], [False, False, True]])

    with torch.no_grad():
        encoded_a, pooled_a = encoder(tokens_a, key_padding_mask=padding_mask)
        encoded_b, pooled_b = encoder(tokens_b, key_padding_mask=padding_mask)

    torch.testing.assert_close(encoded_a[1, :2, :], encoded_b[1, :2, :])
    torch.testing.assert_close(pooled_a[1], pooled_b[1])


def test_embedding_collapse_diagnostics_flags_a_collapsed_batch():
    collapsed = torch.ones(10, 16) * 3.0  # every row identical -> fully collapsed
    diagnostics = embedding_collapse_diagnostics(collapsed)
    assert diagnostics["effective_rank"] < 1.5
    assert diagnostics["per_dim_std_min"] < 1e-6
    violations = check_collapse_gate(diagnostics)
    assert violations


def test_embedding_collapse_diagnostics_passes_a_healthy_batch():
    torch.manual_seed(0)
    healthy = torch.randn(200, 16)  # isotropic noise, well-spread
    diagnostics = embedding_collapse_diagnostics(healthy)
    assert diagnostics["effective_rank"] > 8.0
    violations = check_collapse_gate(diagnostics)
    assert violations == []


def test_embedding_collapse_diagnostics_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        embedding_collapse_diagnostics(torch.randn(1, 16))

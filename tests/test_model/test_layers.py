"""Tests for proteinmpnn.model.layers module."""

from __future__ import annotations

import pytest
import torch

from proteinmpnn.model.layers import (
    DecLayer,
    EncLayer,
    PositionalEncodings,
    PositionWiseFeedForward,
    ProteinFeatures,
)


class TestPositionWiseFeedForward:
    """Tests for PositionWiseFeedForward module."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input shape."""
        num_hidden, num_ff = 64, 256
        B, L = 2, 10
        layer = PositionWiseFeedForward(num_hidden, num_ff)
        x = torch.randn(B, L, num_hidden)

        result = layer(x)

        assert result.shape == (B, L, num_hidden)

    def test_different_batch_sizes(self) -> None:
        """Test with various batch sizes."""
        num_hidden, num_ff = 32, 128
        layer = PositionWiseFeedForward(num_hidden, num_ff)

        for B in [1, 4, 16]:
            x = torch.randn(B, 5, num_hidden)
            result = layer(x)
            assert result.shape == (B, 5, num_hidden)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the layer."""
        layer = PositionWiseFeedForward(16, 64)
        x = torch.randn(1, 4, 16, requires_grad=True)

        result = layer(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestPositionalEncodings:
    """Tests for PositionalEncodings module."""

    def test_output_shape(self) -> None:
        """Test output shape is [B, L, K, num_embeddings]."""
        num_embeddings = 16
        B, L, K = 2, 10, 5
        layer = PositionalEncodings(num_embeddings)
        offset = torch.randint(-10, 10, (B, L, K))
        mask = torch.ones(B, L, K)

        result = layer(offset, mask)

        assert result.shape == (B, L, K, num_embeddings)

    def test_with_different_max_relative(self) -> None:
        """Test with different max_relative_feature values."""
        for max_rel in [16, 32, 64]:
            layer = PositionalEncodings(8, max_relative_feature=max_rel)
            offset = torch.randint(-max_rel, max_rel, (1, 4, 3))
            mask = torch.ones(1, 4, 3)

            result = layer(offset, mask)
            assert result.shape == (1, 4, 3, 8)

    def test_mask_effect(self) -> None:
        """Test that mask affects output differently from no mask."""
        layer = PositionalEncodings(16)
        offset = torch.tensor([[[5, -5]]])
        mask_ones = torch.ones(1, 1, 2)
        mask_zeros = torch.zeros(1, 1, 2)

        result_masked = layer(offset, mask_ones)
        result_unmasked = layer(offset, mask_zeros)

        # Results should differ when mask is different
        assert not torch.allclose(result_masked, result_unmasked)


class TestEncLayer:
    """Tests for EncLayer module."""

    def test_output_shapes(self) -> None:
        """Test that output shapes match input shapes."""
        num_hidden = 64
        B, L, K = 2, 10, 5
        layer = EncLayer(num_hidden, num_hidden * 2)

        h_V = torch.randn(B, L, num_hidden)
        h_E = torch.randn(B, L, K, num_hidden)
        E_idx = torch.randint(0, L, (B, L, K))
        mask_V = torch.ones(B, L)
        mask_attend = torch.ones(B, L, K)

        h_V_out, h_E_out = layer(h_V, h_E, E_idx, mask_V, mask_attend)

        assert h_V_out.shape == h_V.shape
        assert h_E_out.shape == h_E.shape

    def test_without_masks(self) -> None:
        """Test that layer works without masks."""
        num_hidden = 32
        B, L, K = 1, 5, 3
        layer = EncLayer(num_hidden, num_hidden * 2)

        h_V = torch.randn(B, L, num_hidden)
        h_E = torch.randn(B, L, K, num_hidden)
        E_idx = torch.randint(0, L, (B, L, K))

        h_V_out, h_E_out = layer(h_V, h_E, E_idx)

        assert h_V_out.shape == h_V.shape
        assert h_E_out.shape == h_E.shape

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the layer."""
        num_hidden = 16
        layer = EncLayer(num_hidden, num_hidden * 2)

        h_V = torch.randn(1, 4, num_hidden, requires_grad=True)
        h_E = torch.randn(1, 4, 2, num_hidden, requires_grad=True)
        E_idx = torch.randint(0, 4, (1, 4, 2))

        h_V_out, h_E_out = layer(h_V, h_E, E_idx)
        loss = h_V_out.sum() + h_E_out.sum()
        loss.backward()

        assert h_V.grad is not None
        assert h_E.grad is not None


class TestDecLayer:
    """Tests for DecLayer module."""

    def test_output_shape(self) -> None:
        """Test that output shape matches h_V input shape."""
        num_hidden = 64
        B, L, K = 2, 8, 4
        layer = DecLayer(num_hidden, num_hidden * 3)

        h_V = torch.randn(B, L, num_hidden)
        h_E = torch.randn(B, L, K, num_hidden * 3)
        mask_V = torch.ones(B, L)
        mask_attend = torch.ones(B, L, K)

        h_V_out = layer(h_V, h_E, mask_V, mask_attend)

        assert h_V_out.shape == h_V.shape

    def test_without_masks(self) -> None:
        """Test that layer works without masks."""
        num_hidden = 32
        B, L, K = 1, 6, 3
        layer = DecLayer(num_hidden, num_hidden * 3)

        h_V = torch.randn(B, L, num_hidden)
        h_E = torch.randn(B, L, K, num_hidden * 3)

        h_V_out = layer(h_V, h_E)

        assert h_V_out.shape == h_V.shape

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the layer."""
        num_hidden = 16
        layer = DecLayer(num_hidden, num_hidden * 3)

        h_V = torch.randn(1, 4, num_hidden, requires_grad=True)
        h_E = torch.randn(1, 4, 2, num_hidden * 3, requires_grad=True)

        h_V_out = layer(h_V, h_E)
        loss = h_V_out.sum()
        loss.backward()

        assert h_V.grad is not None
        assert h_E.grad is not None


class TestProteinFeatures:
    """Tests for ProteinFeatures module."""

    def test_output_shapes(self) -> None:
        """Test that outputs have correct shapes."""
        edge_features, node_features = 128, 128
        top_k = 30
        B, L = 1, 50  # L > top_k so we get exactly top_k neighbors
        layer = ProteinFeatures(edge_features, node_features, top_k=top_k)

        # X has backbone coords [N, CA, C, O]
        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        residue_idx = torch.arange(L).unsqueeze(0)
        chain_labels = torch.ones(B, L, dtype=torch.long)

        E, E_idx = layer(X, mask, residue_idx, chain_labels)

        assert E.shape == (B, L, top_k, edge_features)
        assert E_idx.shape == (B, L, top_k)

    def test_with_smaller_sequence(self) -> None:
        """Test with sequence smaller than top_k."""
        edge_features, node_features = 64, 64
        top_k = 30
        B, L = 1, 10  # L < top_k
        layer = ProteinFeatures(edge_features, node_features, top_k=top_k)

        X = torch.randn(B, L, 4, 3)
        mask = torch.ones(B, L)
        residue_idx = torch.arange(L).unsqueeze(0)
        chain_labels = torch.ones(B, L, dtype=torch.long)

        E, E_idx = layer(X, mask, residue_idx, chain_labels)

        # K should be min(top_k, L)
        expected_k = min(top_k, L)
        assert E.shape == (B, L, expected_k, edge_features)
        assert E_idx.shape == (B, L, expected_k)

    def test_augmentation(self) -> None:
        """Test that augmentation adds noise when eps > 0."""
        torch.manual_seed(42)
        layer_no_aug = ProteinFeatures(32, 32, augment_eps=0.0)
        layer_aug = ProteinFeatures(32, 32, augment_eps=0.1)

        X = torch.randn(1, 5, 4, 3)
        mask = torch.ones(1, 5)
        residue_idx = torch.arange(5).unsqueeze(0)
        chain_labels = torch.ones(1, 5, dtype=torch.long)

        # Run multiple times with augmentation
        torch.manual_seed(0)
        E1, _ = layer_aug(X.clone(), mask, residue_idx, chain_labels)
        torch.manual_seed(1)
        E2, _ = layer_aug(X.clone(), mask, residue_idx, chain_labels)

        # Results should differ with different random seeds when augmenting
        assert not torch.allclose(E1, E2)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the layer."""
        layer = ProteinFeatures(32, 32, top_k=5, augment_eps=0.0)

        X = torch.randn(1, 8, 4, 3, requires_grad=True)
        mask = torch.ones(1, 8)
        residue_idx = torch.arange(8).unsqueeze(0)
        chain_labels = torch.ones(1, 8, dtype=torch.long)

        E, E_idx = layer(X, mask, residue_idx, chain_labels)
        loss = E.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.all(X.grad == 0)

    def test_multi_chain(self) -> None:
        """Test with multiple chains."""
        layer = ProteinFeatures(32, 32, top_k=10)

        X = torch.randn(1, 20, 4, 3)
        mask = torch.ones(1, 20)
        residue_idx = torch.arange(20).unsqueeze(0)
        # Two chains: first 10 residues chain 1, next 10 chain 2
        chain_labels = torch.cat([
            torch.ones(1, 10, dtype=torch.long),
            torch.ones(1, 10, dtype=torch.long) * 2,
        ], dim=1)

        E, E_idx = layer(X, mask, residue_idx, chain_labels)

        assert E.shape == (1, 20, 10, 32)
        assert E_idx.shape == (1, 20, 10)

"""Tests for proteinmpnn.model.utils module."""

from __future__ import annotations

import pytest
import torch

from proteinmpnn.model.utils import (
    ALPHABET,
    S_to_seq,
    _S_to_seq,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
)


class TestGatherEdges:
    """Tests for gather_edges function."""

    def test_basic_gather(self) -> None:
        """Test that gather_edges retrieves correct edge features."""
        # Create edge features [B=1, N=3, N=3, C=2]
        edges = torch.arange(18).reshape(1, 3, 3, 2).float()
        # Neighbor indices [B=1, N=3, K=2] - for each node, which neighbors to gather
        neighbor_idx = torch.tensor([[[0, 1], [1, 2], [0, 2]]])

        result = gather_edges(edges, neighbor_idx)

        # Result should be [B=1, N=3, K=2, C=2]
        assert result.shape == (1, 3, 2, 2)
        # For node 0, neighbors are [0, 1], so we get edges[0, 0, [0,1], :]
        torch.testing.assert_close(result[0, 0, 0, :], edges[0, 0, 0, :])
        torch.testing.assert_close(result[0, 0, 1, :], edges[0, 0, 1, :])

    def test_output_shape(self) -> None:
        """Test that output shape is [B, N, K, C]."""
        B, N, C, K = 2, 5, 8, 3
        edges = torch.randn(B, N, N, C)
        neighbor_idx = torch.randint(0, N, (B, N, K))

        result = gather_edges(edges, neighbor_idx)

        assert result.shape == (B, N, K, C)

    def test_batch_independence(self) -> None:
        """Test that batches are processed independently."""
        edges = torch.randn(2, 4, 4, 3)
        neighbor_idx = torch.tensor([
            [[0, 1], [1, 2], [2, 3], [0, 3]],
            [[3, 2], [2, 1], [1, 0], [3, 0]],
        ])

        result = gather_edges(edges, neighbor_idx)

        # Verify batch 0 gathers from edges[0] and batch 1 from edges[1]
        torch.testing.assert_close(result[0, 0, 0, :], edges[0, 0, 0, :])
        torch.testing.assert_close(result[1, 0, 0, :], edges[1, 0, 3, :])


class TestGatherNodes:
    """Tests for gather_nodes function."""

    def test_basic_gather(self) -> None:
        """Test that gather_nodes retrieves correct node features."""
        # Node features [B=1, N=4, C=2]
        nodes = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        # Neighbor indices [B=1, N=4, K=2]
        neighbor_idx = torch.tensor([[[1, 2], [0, 3], [0, 1], [2, 3]]])

        result = gather_nodes(nodes, neighbor_idx)

        # Result should be [B=1, N=4, K=2, C=2]
        assert result.shape == (1, 4, 2, 2)
        # For node 0, neighbors are [1, 2], so we get nodes[0, [1,2], :]
        torch.testing.assert_close(result[0, 0, 0, :], nodes[0, 1, :])
        torch.testing.assert_close(result[0, 0, 1, :], nodes[0, 2, :])

    def test_output_shape(self) -> None:
        """Test that output shape is [B, N, K, C]."""
        B, N, C, K = 2, 6, 4, 3
        nodes = torch.randn(B, N, C)
        neighbor_idx = torch.randint(0, N, (B, N, K))

        result = gather_nodes(nodes, neighbor_idx)

        assert result.shape == (B, N, K, C)


class TestGatherNodesT:
    """Tests for gather_nodes_t function."""

    def test_basic_gather(self) -> None:
        """Test transposed gather at specific indices."""
        # Node features [B=1, N=4, C=2]
        nodes = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        # Indices [B=1, K=2]
        neighbor_idx = torch.tensor([[1, 3]])

        result = gather_nodes_t(nodes, neighbor_idx)

        # Result should be [B=1, K=2, C=2]
        assert result.shape == (1, 2, 2)
        torch.testing.assert_close(result[0, 0, :], nodes[0, 1, :])
        torch.testing.assert_close(result[0, 1, :], nodes[0, 3, :])

    def test_output_shape(self) -> None:
        """Test that output shape is [B, K, C]."""
        B, N, C, K = 2, 8, 5, 4
        nodes = torch.randn(B, N, C)
        neighbor_idx = torch.randint(0, N, (B, K))

        result = gather_nodes_t(nodes, neighbor_idx)

        assert result.shape == (B, K, C)


class TestCatNeighborsNodes:
    """Tests for cat_neighbors_nodes function."""

    def test_concatenation(self) -> None:
        """Test that neighbor nodes and edges are concatenated correctly."""
        B, N, K, C = 1, 3, 2, 4
        h_nodes = torch.randn(B, N, C)
        h_neighbors = torch.randn(B, N, K, C)
        E_idx = torch.randint(0, N, (B, N, K))

        result = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)

        # Result should have doubled channel dimension
        assert result.shape == (B, N, K, 2 * C)

    def test_correct_order(self) -> None:
        """Test that h_neighbors comes before gathered h_nodes."""
        B, N, K, C = 1, 2, 2, 2
        h_nodes = torch.ones(B, N, C) * 2.0
        h_neighbors = torch.ones(B, N, K, C) * 1.0
        E_idx = torch.zeros(B, N, K, dtype=torch.long)  # Always gather from node 0

        result = cat_neighbors_nodes(h_nodes, h_neighbors, E_idx)

        # First half should be h_neighbors (1.0), second half should be h_nodes (2.0)
        torch.testing.assert_close(result[:, :, :, :C], h_neighbors)
        torch.testing.assert_close(
            result[:, :, :, C:], torch.ones(B, N, K, C) * 2.0
        )


class TestSToSeq:
    """Tests for S_to_seq function."""

    def test_basic_conversion(self) -> None:
        """Test conversion of sequence tensor to string."""
        # Indices for A, C, D, E (0, 1, 2, 3)
        S = torch.tensor([0, 1, 2, 3])
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = S_to_seq(S, mask)

        assert result == "ACDE"

    def test_with_mask(self) -> None:
        """Test that mask filters out positions."""
        S = torch.tensor([0, 1, 2, 3])  # A, C, D, E
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Only keep A and D

        result = S_to_seq(S, mask)

        assert result == "AD"

    def test_empty_mask(self) -> None:
        """Test with all-zero mask returns empty string."""
        S = torch.tensor([0, 1, 2, 3])
        mask = torch.tensor([0.0, 0.0, 0.0, 0.0])

        result = S_to_seq(S, mask)

        assert result == ""

    def test_full_alphabet(self) -> None:
        """Test that all amino acids convert correctly."""
        S = torch.arange(21)  # All amino acids including X
        mask = torch.ones(21)

        result = S_to_seq(S, mask)

        assert result == ALPHABET
        assert len(result) == 21

    def test_backward_compat_alias(self) -> None:
        """Test that _S_to_seq is an alias for S_to_seq."""
        S = torch.tensor([0, 5, 10])
        mask = torch.ones(3)

        assert S_to_seq(S, mask) == _S_to_seq(S, mask)


class TestAlphabet:
    """Tests for the ALPHABET constant."""

    def test_alphabet_length(self) -> None:
        """Test that alphabet has 21 characters (20 AA + X)."""
        assert len(ALPHABET) == 21

    def test_alphabet_contents(self) -> None:
        """Test that alphabet contains expected amino acids."""
        assert "A" in ALPHABET
        assert "X" in ALPHABET  # Unknown/padding
        assert "C" in ALPHABET  # Cysteine
        assert "W" in ALPHABET  # Tryptophan

    def test_no_gaps(self) -> None:
        """Test that alphabet doesn't contain gap character."""
        assert "-" not in ALPHABET

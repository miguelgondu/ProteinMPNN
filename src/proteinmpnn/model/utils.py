"""Utility functions for ProteinMPNN model operations.

This module contains graph manipulation utilities for gathering features
at neighbor indices, and sequence conversion utilities.
"""

from __future__ import annotations

import torch

# Amino acid alphabet used by ProteinMPNN
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather edge features at neighbor indices.

    Args:
        edges: Edge features of shape [B, N, N, C].
        neighbor_idx: Neighbor indices of shape [B, N, K].

    Returns:
        Neighbor edge features of shape [B, N, K, C].
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices.

    Args:
        nodes: Node features of shape [B, N, C].
        neighbor_idx: Neighbor indices of shape [B, N, K].

    Returns:
        Neighbor node features of shape [B, N, K, C].
    """
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def gather_nodes_t(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices (transposed version).

    Args:
        nodes: Node features of shape [B, N, C].
        neighbor_idx: Neighbor indices of shape [B, K].

    Returns:
        Neighbor features of shape [B, K, C].
    """
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)


def cat_neighbors_nodes(
    h_nodes: torch.Tensor, h_neighbors: torch.Tensor, E_idx: torch.Tensor
) -> torch.Tensor:
    """Concatenate neighbor node features with edge features.

    Args:
        h_nodes: Node features of shape [B, N, C].
        h_neighbors: Neighbor/edge features of shape [B, N, K, C'].
        E_idx: Edge indices of shape [B, N, K].

    Returns:
        Concatenated features of shape [B, N, K, C + C'].
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)


def S_to_seq(S: torch.Tensor, mask: torch.Tensor) -> str:
    """Convert sequence tensor to amino acid string.

    Args:
        S: Sequence indices tensor of shape [L] with values in [0, 20].
        mask: Binary mask tensor of shape [L] indicating valid positions.

    Returns:
        Amino acid sequence string containing only positions where mask > 0.

    Example:
        >>> S = torch.tensor([0, 1, 2, 3])  # A, C, D, E
        >>> mask = torch.tensor([1, 1, 0, 1])
        >>> S_to_seq(S, mask)
        'ACE'
    """
    return "".join([ALPHABET[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])


# Backward compatibility alias
_S_to_seq = S_to_seq

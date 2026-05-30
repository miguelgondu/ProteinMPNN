"""Loss functions for ProteinMPNN training and scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def scores(
    S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute negative log probability scores for sequences.

    Args:
        S: Target sequence tensor of shape [B, L] with amino acid indices.
        log_probs: Log probability tensor of shape [B, L, vocab_size].
        mask: Binary mask tensor of shape [B, L] indicating valid positions.

    Returns:
        Tuple of:
        - scores: Mean NLL per sequence, shape [B]
        - scores_per_res: NLL per residue, shape [B, L]
    """
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    scores_per_res = loss * mask
    scores_val = torch.sum(scores_per_res, dim=-1) / torch.sum(mask, dim=-1)
    return scores_val, scores_per_res


def loss_nll(
    S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute negative log likelihood loss.

    Args:
        S: Target sequence tensor of shape [B, L] with amino acid indices.
        log_probs: Log probability tensor of shape [B, L, vocab_size].
        mask: Binary mask tensor of shape [B, L] indicating valid positions.

    Returns:
        Tuple of:
        - loss: Per-position NLL, shape [B, L]
        - loss_av: Scalar average NLL over all masked positions.
    """
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(
    S: torch.Tensor,
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute label-smoothed negative log likelihood loss.

    Args:
        S: Target sequence tensor of shape [B, L] with amino acid indices.
        log_probs: Log probability tensor of shape [B, L, vocab_size].
        mask: Binary mask tensor of shape [B, L] indicating valid positions.
        weight: Label smoothing weight. Default is 0.1.

    Returns:
        Tuple of:
        - loss: Per-position smoothed NLL, shape [B, L]
        - loss_av: Scalar average smoothed NLL over all masked positions.
    """
    S_onehot = F.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


# Backward compatibility aliases
_scores = scores

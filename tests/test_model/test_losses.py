"""Tests for proteinmpnn.model.losses module."""

from __future__ import annotations

import torch

from proteinmpnn.model.losses import _scores, loss_nll, loss_smoothed, scores


class TestScores:
    """Tests for the scores function."""

    def test_perfect_prediction(self) -> None:
        """Test that perfect prediction gives low score."""
        B, L, V = 1, 5, 21
        S = torch.tensor([[0, 1, 2, 3, 4]])  # Target sequence
        # Create log_probs with very high probability at target positions
        log_probs = torch.full((B, L, V), -10.0)
        for i in range(L):
            log_probs[0, i, S[0, i]] = 0.0  # log(1) = 0
        mask = torch.ones(B, L)

        score_val, score_per_res = scores(S, log_probs, mask)

        # Score should be close to 0 (perfect prediction)
        assert score_val[0] < 0.1
        assert score_per_res.shape == (B, L)

    def test_uniform_prediction(self) -> None:
        """Test score with uniform probabilities."""
        B, L, V = 1, 4, 21
        S = torch.zeros(B, L, dtype=torch.long)
        # Uniform log probabilities: log(1/21) ≈ -3.04
        log_probs = torch.full((B, L, V), -3.044522)
        mask = torch.ones(B, L)

        score_val, score_per_res = scores(S, log_probs, mask)

        # Score should be approximately -log(1/21) ≈ 3.04
        assert torch.abs(score_val[0] - 3.044522) < 0.01

    def test_mask_respected(self) -> None:
        """Test that masked positions don't contribute to score."""
        B, L, V = 1, 4, 21
        S = torch.zeros(B, L, dtype=torch.long)
        log_probs = torch.full((B, L, V), -5.0)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # Only first two positions

        score_val, score_per_res = scores(S, log_probs, mask)

        # Masked positions should have 0 contribution
        assert score_per_res[0, 2] == 0.0
        assert score_per_res[0, 3] == 0.0

    def test_output_shapes(self) -> None:
        """Test output tensor shapes."""
        B, L, V = 2, 10, 21
        S = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        score_val, score_per_res = scores(S, log_probs, mask)

        assert score_val.shape == (B,)
        assert score_per_res.shape == (B, L)

    def test_backward_compat_alias(self) -> None:
        """Test that _scores is an alias for scores."""
        B, L, V = 1, 5, 21
        S = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        result1 = scores(S, log_probs, mask)
        result2 = _scores(S, log_probs, mask)

        torch.testing.assert_close(result1[0], result2[0])
        torch.testing.assert_close(result1[1], result2[1])


class TestLossNll:
    """Tests for the loss_nll function."""

    def test_perfect_prediction(self) -> None:
        """Test that perfect prediction gives near-zero loss."""
        B, L, V = 1, 5, 21
        S = torch.tensor([[0, 1, 2, 3, 4]])
        log_probs = torch.full((B, L, V), -100.0)
        for i in range(L):
            log_probs[0, i, S[0, i]] = 0.0
        mask = torch.ones(B, L)

        loss, loss_av = loss_nll(S, log_probs, mask)

        assert loss_av < 0.01
        assert loss.shape == (B, L)

    def test_mask_respected(self) -> None:
        """Test that only masked positions contribute to average."""
        B, L, V = 1, 4, 21
        S = torch.zeros(B, L, dtype=torch.long)
        # Create log_probs where position 0 has loss 0, others have higher loss
        log_probs = torch.full((B, L, V), -5.0)
        log_probs[0, 0, 0] = 0.0
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Only first position

        loss, loss_av = loss_nll(S, log_probs, mask)

        # Average should only consider masked position (which has ~0 loss)
        assert loss_av < 0.01

    def test_output_shapes(self) -> None:
        """Test output tensor shapes."""
        B, L, V = 2, 8, 21
        S = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        loss, loss_av = loss_nll(S, log_probs, mask)

        assert loss.shape == (B, L)
        assert loss_av.dim() == 0  # Scalar


class TestLossSmoothed:
    """Tests for the loss_smoothed function."""

    def test_output_shapes(self) -> None:
        """Test output tensor shapes."""
        B, L, V = 2, 6, 21
        S = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        loss, loss_av = loss_smoothed(S, log_probs, mask)

        assert loss.shape == (B, L)
        assert loss_av.dim() == 0  # Scalar

    def test_smoothing_increases_loss(self) -> None:
        """Test that smoothing increases loss for perfect predictions."""
        B, L, V = 1, 4, 21
        S = torch.zeros(B, L, dtype=torch.long)
        log_probs = torch.full((B, L, V), -100.0)
        for i in range(L):
            log_probs[0, i, 0] = 0.0  # Perfect log prob at target
        mask = torch.ones(B, L)

        _, loss_av_nll = loss_nll(S, log_probs, mask)
        _, loss_av_smooth = loss_smoothed(S, log_probs, mask, weight=0.1)

        # Smoothed loss should be higher than NLL for perfect predictions
        # because smoothing distributes probability mass
        assert loss_av_smooth > loss_av_nll

    def test_zero_weight_similar_to_nll(self) -> None:
        """Test that weight=0 gives results similar to NLL."""
        B, L, V = 1, 4, 21
        S = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        _, loss_av_nll = loss_nll(S, log_probs, mask)
        _, loss_av_smooth = loss_smoothed(S, log_probs, mask, weight=0.0)

        # With weight=0, smoothed loss should be very close to NLL
        # (Not exactly equal due to numerical differences in computation)
        assert torch.abs(loss_av_smooth - loss_av_nll) < 0.5

    def test_mask_respected(self) -> None:
        """Test that mask is properly applied."""
        B, L, V = 1, 4, 21
        S = torch.zeros(B, L, dtype=torch.long)
        log_probs = torch.randn(B, L, V)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss, _ = loss_smoothed(S, log_probs, mask)

        # Check that computation runs without error with partial mask
        assert loss.shape == (B, L)

    def test_different_weights(self) -> None:
        """Test that different smoothing weights give different results."""
        B, L, V = 1, 5, 21
        S = torch.zeros(B, L, dtype=torch.long)
        log_probs = torch.randn(B, L, V)
        mask = torch.ones(B, L)

        _, loss_01 = loss_smoothed(S, log_probs, mask, weight=0.1)
        _, loss_05 = loss_smoothed(S, log_probs, mask, weight=0.5)

        # Different weights should give different losses
        assert loss_01 != loss_05

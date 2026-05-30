"""Tests for the compute_probs functionality.

Note: These tests require model weights to be present. Tests will be skipped
if weights are not available.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteinmpnn.inference import InferenceRunner
from proteinmpnn.inference.results import ConditionalProbsResult, ResidueInfo
from proteinmpnn.model.utils import ALPHABET
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestComputeProbs:
    """Tests for compute_probs method."""

    @pytest.fixture
    def runner(self) -> InferenceRunner:
        """Create an InferenceRunner instance."""
        return InferenceRunner(model_name="v_48_020")

    @pytest.fixture
    def pdb_path(self):
        """Get path to test PDB file."""
        pdb = DATA_DIR / "6MRR.pdb"
        if not pdb.exists():
            pytest.skip(f"Test PDB not found: {pdb}")
        return pdb

    def test_conditional_probs_output_shape(self, runner, pdb_path):
        """Test that conditional probs returns correct output shape."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            unconditional=False,
            seed=42,
        )

        assert isinstance(result, ConditionalProbsResult)
        assert result.log_probs.ndim == 2
        assert result.log_probs.shape[1] == 21  # 21 amino acids

    def test_unconditional_probs_output_shape(self, runner, pdb_path):
        """Test that unconditional probs returns correct output shape."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )

        assert isinstance(result, ConditionalProbsResult)
        assert result.log_probs.ndim == 2
        assert result.log_probs.shape[1] == 21

    def test_residue_info_matches_sequence_length(self, runner, pdb_path):
        """Test that residue_info length matches log_probs rows."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            seed=42,
        )

        assert len(result.residue_info) == result.log_probs.shape[0]

    def test_residue_info_structure(self, runner, pdb_path):
        """Test that residue_info contains valid ResidueInfo objects."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=42,
        )

        for res_info in result.residue_info:
            assert isinstance(res_info, ResidueInfo)
            assert isinstance(res_info.chain, str)
            assert len(res_info.chain) == 1
            assert isinstance(res_info.residue_idx, int)
            assert res_info.residue_idx >= 1

    def test_mode_is_correct(self, runner, pdb_path):
        """Test that mode is set correctly based on unconditional flag."""
        result_cond = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=False,
            seed=42,
        )
        assert result_cond.mode == "conditional"

        result_uncond = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )
        assert result_uncond.mode == "unconditional"

    def test_reproducibility_with_seed(self, runner, pdb_path):
        """Test that same seed produces same results."""
        result1 = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=12345,
        )
        result2 = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=12345,
        )

        np.testing.assert_array_almost_equal(result1.log_probs, result2.log_probs)

    def test_log_probs_are_valid(self, runner, pdb_path):
        """Test that log probs are valid (negative or zero, sum to ~1 when exp'd)."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            seed=42,
        )

        # Log probs should be <= 0
        assert np.all(result.log_probs <= 0)

        # Probabilities should sum to approximately 1 for each position
        probs = np.exp(result.log_probs)
        sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(sums)), decimal=5)


class TestConditionalProbsResultMethods:
    """Tests for ConditionalProbsResult to_csv and to_npz_dict methods."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample ConditionalProbsResult for testing."""
        log_probs = np.array(
            [
                [-2.0] * 21,
                [-3.0] * 21,
            ],
            dtype=np.float32,
        )

        residue_info = [
            ResidueInfo(chain="A", residue_idx=1),
            ResidueInfo(chain="A", residue_idx=2),
        ]

        return ConditionalProbsResult(
            protein_name="test_protein",
            model_name="v_48_020",
            log_probs=log_probs,
            residue_info=residue_info,
            mode="conditional",
        )

    def test_to_csv_format(self, sample_result):
        """Test that to_csv produces correct format."""
        csv = sample_result.to_csv()
        lines = csv.split("\n")

        # First line is header comment
        assert lines[0].startswith("# protein=")
        assert "test_protein" in lines[0]
        assert "v_48_020" in lines[0]
        assert "conditional" in lines[0]

        # Second line is column headers
        headers = lines[1].split(",")
        assert headers[0] == "chain"
        assert headers[1] == "residue_idx"
        assert len(headers) == 2 + 21  # chain, residue_idx, + 21 amino acids

        # Check amino acid columns match ALPHABET
        for i, aa in enumerate(ALPHABET):
            assert headers[2 + i] == aa

        # Data rows
        assert len(lines) == 4  # header comment + headers + 2 data rows
        row1 = lines[2].split(",")
        assert row1[0] == "A"
        assert row1[1] == "1"

    def test_to_npz_dict_structure(self, sample_result):
        """Test that to_npz_dict produces correct structure."""
        npz_dict = sample_result.to_npz_dict()

        assert "log_probs" in npz_dict
        assert "residue_info" in npz_dict
        assert "alphabet" in npz_dict
        assert "metadata" in npz_dict

        # Check shapes and types
        assert npz_dict["log_probs"].shape == (2, 21)
        assert len(npz_dict["residue_info"]) == 2
        assert npz_dict["alphabet"] == ALPHABET

        # Check metadata
        assert npz_dict["metadata"]["protein_name"] == "test_protein"
        assert npz_dict["metadata"]["model_name"] == "v_48_020"
        assert npz_dict["metadata"]["mode"] == "conditional"

    def test_to_npz_dict_residue_info_structure(self, sample_result):
        """Test that residue_info in npz_dict is structured array."""
        npz_dict = sample_result.to_npz_dict()
        residue_info = npz_dict["residue_info"]

        # Should be structured array with chain and residue_idx fields
        assert residue_info.dtype.names == ("chain", "residue_idx")
        assert residue_info[0]["chain"] == "A"
        assert residue_info[0]["residue_idx"] == 1

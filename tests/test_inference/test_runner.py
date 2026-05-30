"""Tests for the InferenceRunner.

Note: These tests require model weights to be present. Tests will be skipped
if weights are not available.
"""

from __future__ import annotations

import pytest

from proteinmpnn.inference import InferenceRunner
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"
EXAMPLES_DIR = ROOT_DIR / "examples"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestInferenceRunner:
    """Tests for InferenceRunner class."""

    def test_runner_loads_model(self):
        """Test that the runner successfully loads a model."""
        runner = InferenceRunner(model_name="v_48_020")
        assert runner.model is not None
        assert runner.model_name == "v_48_020"

    def test_invalid_model_name_raises(self):
        """Test that an invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Model weights not found"):
            InferenceRunner(model_name="nonexistent_model")


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestDesignSingle:
    """Tests for design_single method."""

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

    def test_design_single_basic(self, runner, pdb_path):
        """Test basic sequence design."""
        result = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            num_sequences=1,
            seed=42,
        )

        assert result.protein_name == "6MRR"
        assert result.native is not None
        assert len(result.sequences) == 1
        assert result.sequences[0].score > 0

    def test_design_single_multiple_sequences(self, runner, pdb_path):
        """Test generating multiple sequences."""
        result = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            num_sequences=3,
            batch_size=1,
            seed=42,
        )

        assert len(result.sequences) == 3

    def test_design_single_with_temperature(self, runner, pdb_path):
        """Test different temperatures produce different results."""
        result = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            num_sequences=2,
            batch_size=1,
            temperatures=[0.1, 0.5],
            seed=42,
        )

        # With 2 temperatures, num_batches=2, batch_size=1:
        # 2 batches * 2 temps * 1 per batch = 4 sequences
        assert len(result.sequences) == 4
        # Different temperatures should be recorded
        temps = {s.temperature for s in result.sequences}
        assert len(temps) == 2

    def test_design_single_reproducible_with_seed(self, runner, pdb_path):
        """Test that same seed produces same results."""
        result1 = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            num_sequences=1,
            seed=12345,
        )
        result2 = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            num_sequences=1,
            seed=12345,
        )

        assert result1.sequences[0].sequence == result2.sequences[0].sequence
        assert result1.sequences[0].score == result2.sequences[0].score

    def test_design_single_to_fasta(self, runner, pdb_path):
        """Test FASTA output generation."""
        result = runner.design_single(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            num_sequences=2,
            seed=42,
        )

        fasta = result.to_fasta()
        lines = fasta.split("\n")
        # Native + 2 designed = 3 sequences = 6 lines
        assert len(lines) == 6
        # All odd lines should start with >
        for i in range(0, 6, 2):
            assert lines[i].startswith(">")


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestDesignWithSymmetry:
    """Tests for design with symmetric constraints."""

    @pytest.fixture
    def runner(self) -> InferenceRunner:
        """Create an InferenceRunner instance."""
        return InferenceRunner(model_name="v_48_020")

    @pytest.fixture
    def homooligomer_pdb(self):
        """Get path to homooligomer test PDB."""
        pdb = EXAMPLES_DIR / "complex" / "homooligomer" / "4GYT.pdb"
        if not pdb.exists():
            pytest.skip(f"Homooligomer PDB not found: {pdb}")
        return pdb

    def test_design_with_symmetric_residues(self, runner, homooligomer_pdb):
        """Test design with symmetric constraints."""
        result = runner.design_single(
            pdb_path=homooligomer_pdb,
            designable_res="A7-A10,B7-B10",
            symmetric_res="A7-A10:B7-B10",
            num_sequences=1,
            seed=42,
        )

        assert result is not None
        assert len(result.sequences) == 1

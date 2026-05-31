"""Tests for the compute-probs CLI command.

Note: These tests require model weights to be present. Tests will be skipped
if weights are not available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from proteinmpnn.cli import app
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

runner = CliRunner(env={"COLUMNS": "200"})

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestComputeProbsCommand:
    """Tests for compute-probs CLI command."""

    @pytest.fixture
    def pdb_path(self) -> Path:
        """Get path to test PDB file."""
        pdb = DATA_DIR / "6MRR.pdb"
        if not pdb.exists():
            pytest.skip(f"Test PDB not found: {pdb}")
        return pdb

    def test_creates_csv_and_npz_files(self, pdb_path):
        """Test that compute-probs creates both CSV and NPZ files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "compute-probs",
                    str(pdb_path),
                    "--design",
                    "A1-A10",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            csv_path = output_dir / "6MRR_probs.csv"
            npz_path = output_dir / "6MRR_probs.npz"

            assert csv_path.exists(), f"CSV file not created: {csv_path}"
            assert npz_path.exists(), f"NPZ file not created: {npz_path}"

    def test_csv_file_content(self, pdb_path):
        """Test that CSV file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "compute-probs",
                    str(pdb_path),
                    "--design",
                    "A1-A5",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0

            csv_path = output_dir / "6MRR_probs.csv"
            content = csv_path.read_text()
            lines = content.strip().split("\n")

            # First line should be header comment
            assert lines[0].startswith("# protein=")
            assert "6MRR" in lines[0]
            assert "conditional" in lines[0]

            # Second line should be column headers
            headers = lines[1].split(",")
            assert headers[0] == "chain"
            assert headers[1] == "residue_idx"
            assert len(headers) == 23  # chain, residue_idx + 21 amino acids

    def test_npz_file_content(self, pdb_path):
        """Test that NPZ file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "compute-probs",
                    str(pdb_path),
                    "--design",
                    "A1-A5",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0

            npz_path = output_dir / "6MRR_probs.npz"
            data = np.load(npz_path, allow_pickle=True)

            assert "log_probs" in data
            assert "residue_info" in data
            assert "alphabet" in data
            assert "metadata" in data

            # Check log_probs shape
            log_probs = data["log_probs"]
            assert log_probs.ndim == 2
            assert log_probs.shape[1] == 21

    def test_unconditional_flag(self, pdb_path):
        """Test that --unconditional flag works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "compute-probs",
                    str(pdb_path),
                    "--unconditional",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Check that mode is unconditional in CSV
            csv_path = output_dir / "6MRR_probs.csv"
            content = csv_path.read_text()
            assert "mode=unconditional" in content

    def test_with_design_flag(self, pdb_path):
        """Test that --design flag works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "compute-probs",
                    str(pdb_path),
                    "--design",
                    "A1-A20",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

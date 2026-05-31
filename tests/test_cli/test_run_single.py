"""Tests for the run_single CLI command.

Note: These tests require model weights to be present. Tests will be skipped
if weights are not available.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from proteinmpnn.cli import app
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

runner = CliRunner()

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestRunSingleCommand:
    """Tests for run-single CLI command.

    Note: Since run_single is the only command, it becomes the default
    command and doesn't require a subcommand name.
    """

    @pytest.fixture
    def pdb_path(self) -> Path:
        """Get path to test PDB file."""
        pdb = DATA_DIR / "6MRR.pdb"
        if not pdb.exists():
            pytest.skip(f"Test PDB not found: {pdb}")
        return pdb

    def test_run_single_creates_fasta(self, pdb_path):
        """Test that run-single creates a FASTA file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "run-single",
                    str(pdb_path),
                    "--design",
                    "A1-A10",
                    "-n",
                    "2",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            fasta_path = output_dir / "6MRR.fasta"
            assert fasta_path.exists(), f"FASTA file not created: {fasta_path}"

            # Check FASTA content
            content = fasta_path.read_text()
            lines = content.strip().split("\n")
            # Native + 2 designed = 3 sequences = 6 lines
            assert len(lines) == 6

    def test_run_single_with_af2_flag(self, pdb_path):
        """Test that --af2 flag creates CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "run-single",
                    str(pdb_path),
                    "--design",
                    "A1-A10",
                    "-n",
                    "1",
                    "--output",
                    str(output_dir),
                    "--af2",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"
            csv_path = output_dir / "6MRR.csv"
            assert csv_path.exists(), f"CSV file not created: {csv_path}"

    def test_run_single_prints_scores(self, pdb_path):
        """Test that run-single prints score information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "run-single",
                    str(pdb_path),
                    "--design",
                    "A1-A10",
                    "-n",
                    "1",
                    "--output",
                    str(output_dir),
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0

    def test_run_single_with_temperature(self, pdb_path):
        """Test run-single with custom temperature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = runner.invoke(
                app,
                [
                    "run-single",
                    str(pdb_path),
                    "--design",
                    "A1-A5",
                    "-n",
                    "1",
                    "--output",
                    str(output_dir),
                    "--temp",
                    "0.2",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_run_single_help(self):
        """Test that --help works."""
        result = runner.invoke(app, ["run-single", "--help"])
        assert result.exit_code == 0
        assert "Design sequences for a single protein structure" in result.output

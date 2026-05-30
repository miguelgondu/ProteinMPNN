"""Tests for SingleStateDesignInput class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proteinmpnn.data.config import SingleStateConfig
from proteinmpnn.data.single_state import SingleStateDesignInput
from proteinmpnn.utils.constants import ROOT_DIR

# Path to test PDB file (go up from tests/test_data/ to repo root)
TEST_PDB = ROOT_DIR / "data" / "pdbs" / "6MRR.pdb"


@pytest.fixture
def pdb_path() -> Path:
    """Return path to test PDB file."""
    if not TEST_PDB.exists():
        pytest.skip(f"Test PDB not found: {TEST_PDB}")
    return TEST_PDB


class TestSingleStateDesignInputInit:
    """Tests for SingleStateDesignInput initialization."""

    def test_basic_init(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(pdb_path=pdb_path)
        assert design.pdb_path == pdb_path
        assert len(design._chain_seqs) > 0

    def test_with_designable_residues(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15",
        )
        assert len(design._design_res) == 6
        assert ("A", 10) in design._design_res
        assert ("A", 15) in design._design_res

    def test_with_symmetric_residues(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15,A20-A25",
            symmetric_res="A10:A20,A11:A21",
        )
        assert len(design._symmetric_res) == 2

    def test_default_design_setting(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            default_design_setting="ACDEF",
        )
        assert design.default_design_setting == "ACDEF"

    def test_invalid_pdb_path_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            SingleStateDesignInput(pdb_path=Path("/nonexistent/path.pdb"))

    def test_invalid_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not a pdb")
        with pytest.raises(ValueError, match="Expected .pdb file"):
            SingleStateDesignInput(pdb_path=bad_file)


class TestSingleStateDesignInputToConfig:
    """Tests for to_config method."""

    def test_returns_single_state_config(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15",
        )
        config = design.to_config()
        assert isinstance(config, SingleStateConfig)

    def test_config_has_sequences(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(pdb_path=pdb_path)
        config = design.to_config()
        assert len(config.sequence) > 0
        assert all(isinstance(seq, str) for seq in config.sequence.values())

    def test_config_has_designable_residues(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15",
        )
        config = design.to_config()
        assert len(config.designable) > 0

    def test_designable_residue_attributes(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10",
        )
        config = design.to_config()
        assert len(config.designable) == 1
        res = config.designable[0]
        assert res.chain == "A"
        assert res.resid == 10
        assert res.WTAA  # Should have wild-type AA
        assert res.MutTo == "all"

    def test_config_has_symmetric_groups(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15,A20-A25",
            symmetric_res="A10:A20,A11:A21",
        )
        config = design.to_config()
        assert len(config.symmetric) == 2


class TestSingleStateDesignInputGenerateJson:
    """Tests for generate_json method."""

    def test_generates_valid_json(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A15",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)

        try:
            design.generate_json(out_path)
            assert out_path.exists()

            with out_path.open() as f:
                data = json.load(f)

            assert "sequence" in data
            assert "designable" in data
            assert "symmetric" in data
        finally:
            out_path.unlink()

    def test_json_has_correct_structure(self, pdb_path: Path) -> None:
        design = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A10-A12",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)

        try:
            design.generate_json(out_path)

            with out_path.open() as f:
                data = json.load(f)

            # Check designable structure
            for res in data["designable"]:
                assert "chain" in res
                assert "resid" in res
                assert "WTAA" in res
                assert "MutTo" in res
        finally:
            out_path.unlink()

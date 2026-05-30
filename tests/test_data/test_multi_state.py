"""Tests for MultiStateDesignInput class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from proteinmpnn.data.config import MultiStateConfig
from proteinmpnn.data.multi_state import MultiStateDesignInput
from proteinmpnn.utils.constants import ROOT_DIR

# Path to test multi-state PDB directory (go up from tests/test_data/ to repo root)
TEST_MSD_DIR = ROOT_DIR / "examples" / "multi_state" / "bidirectional"


@pytest.fixture
def msd_dir() -> Path:
    """Return path to test multi-state PDB directory."""
    if not TEST_MSD_DIR.exists():
        pytest.skip(f"Test MSD directory not found: {TEST_MSD_DIR}")
    pdb_files = list(TEST_MSD_DIR.glob("*.pdb"))
    # Filter out msd subdirectory files
    pdb_files = [p for p in pdb_files if "msd" not in str(p)]
    if len(pdb_files) < 2:
        pytest.skip("Need at least 2 PDB files for multi-state tests")
    return TEST_MSD_DIR


@pytest.fixture
def pdb_names(msd_dir: Path) -> list[str]:
    """Return sorted list of PDB stems in the test directory."""
    pdb_files = sorted(msd_dir.glob("*.pdb"))
    return [p.stem for p in pdb_files if "msd" not in str(p)]


class TestMultiStateDesignInputInit:
    """Tests for MultiStateDesignInput initialization."""

    def test_basic_init(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        assert design.pdb_dir.name == "msd"  # Should point to msd subdir
        assert len(design.chain_dict) == 2

    def test_chain_remapping(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        # First PDB keeps original chain, second gets remapped
        assert pdb_names[0] in design.chain_dict
        assert pdb_names[1] in design.chain_dict

    def test_beta_dict_populated(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        assert len(design._beta_dict) == 2
        # Check beta values are correct
        betas = list(design._beta_dict.values())
        assert 1.0 in betas
        assert 0.5 in betas

    def test_invalid_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            MultiStateDesignInput(
                pdb_dir=Path("/nonexistent/dir"),
                constraints="dummy:A1:1.0,dummy2:A1:0.5",
            )

    def test_insufficient_pdbs_raises(self, tmp_path: Path) -> None:
        # Create directory with only one PDB
        pdb = tmp_path / "single.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00")
        with pytest.raises(ValueError, match="at least 2 PDB files"):
            MultiStateDesignInput(
                pdb_dir=tmp_path,
                constraints="single:A1:1.0,other:A1:0.5",
            )


class TestMultiStateDesignInputToConfig:
    """Tests for to_config method."""

    def test_returns_multi_state_config(
        self, msd_dir: Path, pdb_names: list[str]
    ) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        config = design.to_config()
        assert isinstance(config, MultiStateConfig)

    def test_config_has_tied_betas(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        config = design.to_config()
        assert len(config.tied_betas) > 0

    def test_config_has_chain_key(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
        )
        config = design.to_config()
        assert len(config.chain_key) == 2
        assert pdb_names[0] in config.chain_key
        assert pdb_names[1] in config.chain_key


class TestMultiStateDesignInputGenerateJson:
    """Tests for generate_json method."""

    def test_generates_valid_json(self, msd_dir: Path, pdb_names: list[str]) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
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
            assert "tied_betas" in data
            assert "chain_key" in data
        finally:
            out_path.unlink()


class TestMultiStateBidirectional:
    """Tests for bidirectional symmetry."""

    def test_bidirectional_with_dimeric_constraints(
        self, msd_dir: Path, pdb_names: list[str]
    ) -> None:
        constraints = f"{pdb_names[0]}:B10:1.0,{pdb_names[1]}:B10:0.5"
        design = MultiStateDesignInput(
            pdb_dir=msd_dir,
            constraints=constraints,
            bidirectional=True,
        )
        # Should not raise for dimeric constraints
        assert design.bidirectional is True

"""Tests for factory function and backward compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from proteinmpnn.data.input import ProteinDesignInputFormatter, create_design_input
from proteinmpnn.data.multi_state import MultiStateDesignInput
from proteinmpnn.data.single_state import SingleStateDesignInput
from proteinmpnn.utils.constants import ROOT_DIR

if TYPE_CHECKING:
    from pathlib import Path

# Path to test files (go up from tests/test_data/ to repo root)
TEST_PDB_DIR = ROOT_DIR / "data" / "pdbs"
TEST_MSD_DIR = ROOT_DIR / "examples" / "multi_state" / "bidirectional"


@pytest.fixture
def single_pdb_dir() -> Path:
    """Return path to directory with single PDB file."""
    if not TEST_PDB_DIR.exists():
        pytest.skip(f"Test PDB directory not found: {TEST_PDB_DIR}")
    pdb_files = list(TEST_PDB_DIR.glob("*.pdb"))
    if len(pdb_files) != 1:
        pytest.skip("Expected exactly 1 PDB file in test directory")
    return TEST_PDB_DIR


@pytest.fixture
def multi_pdb_dir() -> Path:
    """Return path to directory with multiple PDB files."""
    if not TEST_MSD_DIR.exists():
        pytest.skip(f"Test MSD directory not found: {TEST_MSD_DIR}")
    return TEST_MSD_DIR


@pytest.fixture
def msd_pdb_names(multi_pdb_dir: Path) -> list[str]:
    """Return sorted list of PDB stems in the test directory."""
    pdb_files = sorted(multi_pdb_dir.glob("*.pdb"))
    return [p.stem for p in pdb_files if "msd" not in str(p)]


class TestCreateDesignInputSingleState:
    """Tests for create_design_input in single-state mode."""

    def test_returns_single_state_input(self, single_pdb_dir: Path) -> None:
        design = create_design_input(pdb_dir=single_pdb_dir)
        assert isinstance(design, SingleStateDesignInput)

    def test_with_designable_residues(self, single_pdb_dir: Path) -> None:
        design = create_design_input(
            pdb_dir=single_pdb_dir,
            designable_res="A10-A15",
        )
        assert isinstance(design, SingleStateDesignInput)
        assert len(design._design_res) == 6

    def test_with_symmetric_residues(self, single_pdb_dir: Path) -> None:
        design = create_design_input(
            pdb_dir=single_pdb_dir,
            designable_res="A10-A15,A20-A25",
            symmetric_res="A10:A20",
        )
        assert isinstance(design, SingleStateDesignInput)
        assert len(design._symmetric_res) == 1


class TestCreateDesignInputMultiState:
    """Tests for create_design_input in multi-state mode."""

    def test_returns_multi_state_input(
        self, multi_pdb_dir: Path, msd_pdb_names: list[str]
    ) -> None:
        constraints = f"{msd_pdb_names[0]}:B10:1.0,{msd_pdb_names[1]}:B10:0.5"
        design = create_design_input(
            pdb_dir=multi_pdb_dir,
            multi_state=True,
            constraints=constraints,
        )
        assert isinstance(design, MultiStateDesignInput)

    def test_multi_state_requires_constraints(self, multi_pdb_dir: Path) -> None:
        with pytest.raises(ValueError, match="without specifying constraints"):
            create_design_input(
                pdb_dir=multi_pdb_dir,
                multi_state=True,
            )


class TestCreateDesignInputValidation:
    """Tests for argument validation."""

    def test_bidirectional_requires_multi_state(self, single_pdb_dir: Path) -> None:
        with pytest.raises(ValueError, match="without enabling multi_state"):
            create_design_input(
                pdb_dir=single_pdb_dir,
                bidirectional=True,
            )

    def test_constraints_requires_multi_state(self, single_pdb_dir: Path) -> None:
        with pytest.raises(ValueError, match="without enabling multi_state"):
            create_design_input(
                pdb_dir=single_pdb_dir,
                constraints="dummy:A1:1.0",
            )

    def test_cannot_mix_constraints_and_symmetric_res(
        self, multi_pdb_dir: Path, msd_pdb_names: list[str]
    ) -> None:
        constraints = f"{msd_pdb_names[0]}:B10:1.0,{msd_pdb_names[1]}:B10:0.5"
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_design_input(
                pdb_dir=multi_pdb_dir,
                multi_state=True,
                constraints=constraints,
                symmetric_res="A1:B1",
            )

    def test_single_state_with_multiple_pdbs_raises(self, multi_pdb_dir: Path) -> None:
        with pytest.raises(ValueError, match="requires 1 PDB file"):
            create_design_input(
                pdb_dir=multi_pdb_dir,
                multi_state=False,
            )

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No PDB files found"):
            create_design_input(pdb_dir=tmp_path)


class TestBackwardCompatibilityAlias:
    """Tests for ProteinDesignInputFormatter backward compatibility."""

    def test_alias_is_factory_function(self) -> None:
        assert ProteinDesignInputFormatter is create_design_input

    def test_alias_works_for_single_state(self, single_pdb_dir: Path) -> None:
        design = ProteinDesignInputFormatter(
            pdb_dir=single_pdb_dir,
            designable_res="A10-A15",
        )
        assert isinstance(design, SingleStateDesignInput)

    def test_alias_works_for_multi_state(
        self, multi_pdb_dir: Path, msd_pdb_names: list[str]
    ) -> None:
        constraints = f"{msd_pdb_names[0]}:B10:1.0,{msd_pdb_names[1]}:B10:0.5"
        design = ProteinDesignInputFormatter(
            pdb_dir=multi_pdb_dir,
            multi_state=True,
            constraints=constraints,
        )
        assert isinstance(design, MultiStateDesignInput)

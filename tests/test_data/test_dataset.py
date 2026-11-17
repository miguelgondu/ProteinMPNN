"""Tests for StructureDatasetPDB loading."""

from __future__ import annotations

import pytest

from proteinmpnn.data.structure.dataset import StructureDatasetPDB
from proteinmpnn.utils.constants import ROOT_DIR

DATA_DIR = ROOT_DIR / "data" / "pdbs"


def _tiny_pdb(name: str, chains: dict[str, list[str]]) -> str:
    """A minimal PDB with one N/CA/C/O residue per entry of each chain."""
    lines = [f"HEADER    {name}"]
    serial = 1
    for chain, seq3 in chains.items():
        for i, resname in enumerate(seq3):
            for atom, dx in (("N", -1.0), ("CA", 0.0), ("C", 1.0), ("O", 1.5)):
                lines.append(
                    f"ATOM  {serial:>5}  {atom:<3} {resname:>3} {chain}{i + 1:>4}    "
                    f"{3.8 * i + dx:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
                    f"          {atom[0]:>2}"
                )
                serial += 1
    lines += ["END", ""]
    return "\n".join(lines)


@pytest.fixture
def pdb_dir(tmp_path):
    """A directory with two structures, so cross-contamination is detectable."""
    (tmp_path / "wanted.pdb").write_text(
        _tiny_pdb("WANTED", {"A": ["ALA", "GLY"], "B": ["TRP"]})
    )
    (tmp_path / "unwanted.pdb").write_text(_tiny_pdb("UNWANTED", {"A": ["LYS"] * 5}))
    return tmp_path


class TestFromPdbFile:
    def test_parses_only_the_given_file(self, pdb_dir):
        dataset = StructureDatasetPDB.from_pdb_file(pdb_dir / "wanted.pdb")

        assert len(dataset) == 1
        assert dataset[0]["name"] == "wanted"
        assert dataset[0]["seq"] == "AGW"

    def test_from_pdb_dir_sees_the_sibling(self, pdb_dir):
        """Contrast: this is what from_pdb_dir does, and why from_pdb_file exists."""
        dataset = StructureDatasetPDB.from_pdb_dir(pdb_dir)

        assert len(dataset) == 2

    def test_restricts_to_requested_chains(self, pdb_dir):
        dataset = StructureDatasetPDB.from_pdb_file(
            pdb_dir / "wanted.pdb", chains=["B"]
        )

        assert dataset[0]["seq"] == "W"
        assert dataset[0]["num_of_chains"] == 1
        assert "seq_chain_A" not in dataset[0]

    def test_missing_chain_lists_available_chains(self, pdb_dir):
        with pytest.raises(ValueError, match=r"\['A', 'B'\]"):
            StructureDatasetPDB.from_pdb_file(pdb_dir / "wanted.pdb", chains=["Z"])

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="Could not find pdb file"):
            StructureDatasetPDB.from_pdb_file(tmp_path / "nope.pdb")

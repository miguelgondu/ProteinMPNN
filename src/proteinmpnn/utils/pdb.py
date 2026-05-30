"""PDB manipulation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from Bio.PDB import PDBParser, Select

if TYPE_CHECKING:
    from Bio.PDB import Atom, Structure


class NotDisordered(Select):
    """Selects non-disordered atoms, preferring altloc 'A'."""

    def accept_atom(self, atom: Atom) -> bool:
        """Accept non-disordered atoms or those with altloc 'A'."""
        return not atom.is_disordered() or atom.get_altloc() == "A"


def check_structure_bounds(structure: Structure) -> None:
    """Validate structure fits within PDB format coordinate limits.

    PDB format has coordinate limits of +/- 9999.999 Angstroms.

    Args:
        structure: BioPython Structure object to validate.

    Raises:
        ValueError: If any atom coordinates exceed PDB format limits.
    """
    for atom in structure.get_atoms():
        coords = atom.get_coord()
        if np.any(np.abs(coords) > 9999.0):
            raise ValueError(
                f"Structure contains coordinates outside PDB limits (+/- 9999 Å): "
                f"atom {atom.get_full_id()} at {coords}"
            )


def calculate_min_inter_state_distance(
    structure: Structure, chain_dict: dict[str, dict[str, str]]
) -> float:
    """Calculate minimum distance between states in multi-state structure.

    This checks that different states (PDB files) are sufficiently separated
    after being combined into a single structure.

    Args:
        structure: Combined BioPython Structure object.
        chain_dict: Mapping from PDB name -> original chain -> remapped chain.

    Returns:
        The minimum distance between any two atoms from different states.
    """
    # Get list of chains per state
    states = list(chain_dict.values())
    if len(states) < 2:
        return float("inf")

    min_dist = float("inf")

    # Get CA atoms grouped by state
    state_atoms: list[list[np.ndarray]] = []
    for state_chains in states:
        remapped_chains = set(state_chains.values())
        atoms = []
        for chain in structure.get_chains():
            if chain.id in remapped_chains:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            atoms.append(atom.get_coord())
        state_atoms.append(atoms)

    # Compare atoms between different states
    for i, atoms_i in enumerate(state_atoms):
        for j, atoms_j in enumerate(state_atoms):
            if i >= j:
                continue
            for coord_i in atoms_i:
                for coord_j in atoms_j:
                    dist = np.sqrt(np.sum((coord_i - coord_j) ** 2))
                    if dist < min_dist:
                        min_dist = dist

    return min_dist


def get_neighbors_within_radius(
    pdb_path: Path,
    center: tuple[str, int],
    radius: float,
) -> list[tuple[str, int]]:
    """Find all residues with CA within radius of center residue.

    Args:
        pdb_path: Path to the PDB file.
        center: Tuple of (chain_id, residue_number) for the center residue.
        radius: Radius in Angstroms to search within.

    Returns:
        List of (chain_id, residue_number) tuples for all residues within the radius.

    Raises:
        ValueError: If the center residue or its CA atom cannot be found.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)

    center_chain, center_res = center
    center_pos = None

    # Find the center CA position
    for chain in structure.get_chains():
        if chain.id == center_chain:
            for residue in chain.get_residues():
                if residue.id[1] == center_res:
                    for atom in residue.get_atoms():
                        if atom.get_name() == "CA":
                            center_pos = atom.get_coord()
                            break
                    break
            break

    if center_pos is None:
        raise ValueError(
            f"Could not find CA for {center_chain}{center_res} in {pdb_path}"
        )

    # Find all neighbors within radius
    neighbors: list[tuple[str, int]] = []
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            res_chain, res_num = chain.id, residue.id[1]
            for atom in residue.get_atoms():
                if atom.get_name() == "CA":
                    dist = np.sqrt(np.sum((atom.get_coord() - center_pos) ** 2) + 1e-12)
                    if dist <= radius:
                        neighbors.append((res_chain, res_num))
                    break

    return neighbors

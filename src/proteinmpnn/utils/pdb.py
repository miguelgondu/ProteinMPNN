"""PDB manipulation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from Bio.PDB.PDBIO import Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

from proteinmpnn.utils.constants import CHAIN_ALPHABET

if TYPE_CHECKING:
    from Bio.PDB.Atom import Atom


class NotDisordered(Select):
    """Selects non-disordered atoms, preferring altloc 'A'."""

    def accept_atom(self, atom: Atom) -> bool:  # type: ignore
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

    if not isinstance(structure, Structure):
        raise ValueError(f"Couldn't parse structure in {pdb_path.stem}")

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


def parse_PDB_biounits(x: str | Path, atoms=["N", "CA", "C"], chain=None):
    """
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    if not isinstance(x, Path):
        x = Path(x)

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    alpha_3 = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]

    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = dict(enumerate(alpha_1))

    def N_to_AA(x_):
        # [[0,1,2,3]] -> ["ARND"]
        x_ = np.array(x_)
        if x_.ndim == 1:
            x_ = x_[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x_]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in x.open("rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x_, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                #         resn = int(resn)
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x_, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(int(min_resn), int(max_resn + 1)):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return "no_chain", "no_chain"


def parse_PDB(path_to_pdb, input_chain_list=None):
    c = 0
    pdb_dict_list = []
    chain_alphabet = CHAIN_ALPHABET

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ""
        for letter in chain_alphabet:
            xyz, seq = parse_PDB_biounits(
                biounit, atoms=["N", "CA", "C", "O"], chain=letter
            )
            if not isinstance(xyz, str):
                concat_seq += seq[0]
                my_dict["seq_chain_" + letter] = seq[0]
                coords_dict_chain = {}
                coords_dict_chain["N_chain_" + letter] = xyz[:, 0, :].tolist()
                coords_dict_chain["CA_chain_" + letter] = xyz[:, 1, :].tolist()
                coords_dict_chain["C_chain_" + letter] = xyz[:, 2, :].tolist()
                coords_dict_chain["O_chain_" + letter] = xyz[:, 3, :].tolist()
                my_dict["coords_chain_" + letter] = coords_dict_chain
                s += 1
        my_dict["name"] = biounit.stem
        my_dict["num_of_chains"] = s
        my_dict["seq"] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list

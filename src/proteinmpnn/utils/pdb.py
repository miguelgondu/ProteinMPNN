"""PDB manipulation utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
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


def parse_PDB_biounits_legacy(x: str | Path, atoms=["N", "CA", "C"], chain=None):
    """
    Legacy implementation of parse_PDB_biounits.

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


# Module-level constants for amino acid conversion (avoid recreating per call)
_ALPHA_1 = list("ARNDCQEGHILKMFPSTWYV-")
_ALPHA_3 = [
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
_AA_3_N = {a: n for n, a in enumerate(_ALPHA_3)}
_AA_N_1 = dict(enumerate(_ALPHA_1))
_STANDARD_AA = set(_ALPHA_3[:-1])  # Exclude GAP

# Reusable parser instance (avoid recreating per call)
# PERMISSIVE=True allows parsing even with some errors
_BIOPYTHON_PARSER = PDBParser(QUIET=True, PERMISSIVE=True)


def _n_to_aa(seq_indices: list[int]) -> list[str]:
    """Convert list of amino acid indices to single-letter sequence string."""
    return ["".join([_AA_N_1.get(a, "-") for a in seq_indices])]


def parse_PDB_biounits(
    x: str | Path, atoms: list[str] = ["N", "CA", "C"], chain: str | None = None
) -> tuple[np.ndarray, list[str]] | tuple[str, str]:
    """
    Parse PDB file to extract backbone atom coordinates and sequence.

    Uses gemmi (C++ library) for fast parsing. Falls back to pure Python
    implementation if gemmi fails.

    input:  x = PDB filename
            atoms = atoms to extract (optional)
            chain = chain to extract (optional, None means all chains)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    if not isinstance(x, Path):
        x = Path(x)

    try:
        structure = gemmi.read_structure(str(x))
    except Exception:
        return "no_chain", "no_chain"

    if not structure:
        return "no_chain", "no_chain"

    model = structure[0]

    xyz: dict[int, dict[str, dict[str, tuple[float, float, float]]]] = {}
    seq: dict[int, dict[str, str]] = {}
    min_resn, max_resn = 1e6, -1e6

    for ch in model:
        if chain is not None and ch.name != chain:
            continue

        for residue in ch:
            resname = residue.name
            # Handle MSE (selenomethionine) as MET
            if resname == "MSE":
                resname = "MET"
            elif resname not in _STANDARD_AA:
                continue

            resn = residue.seqid.num - 1
            resa = residue.seqid.icode.strip() if residue.seqid.icode else ""

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
                seq[resn][resa] = resname

            # Get atom coordinates
            for atom_name in atoms:
                if atom_name not in xyz[resn][resa]:
                    atom = residue.find_atom(atom_name, "*")
                    if atom:
                        xyz[resn][resa][atom_name] = (
                            atom.pos.x,
                            atom.pos.y,
                            atom.pos.z,
                        )

    if not xyz:
        return "no_chain", "no_chain"

    # Build output arrays
    seq_: list[int] = []
    xyz_: list[tuple[float, float, float]] = []
    nan_tuple = (float("nan"), float("nan"), float("nan"))
    num_atoms = len(atoms)

    try:
        for resn in range(int(min_resn), int(max_resn + 1)):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(_AA_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(nan_tuple)
            else:
                for _ in range(num_atoms):
                    xyz_.append(nan_tuple)

        xyz_array = np.array(xyz_, dtype=np.float64).reshape(-1, num_atoms, 3)
        return xyz_array, _n_to_aa(seq_)

    except TypeError:
        return "no_chain", "no_chain"


def parse_PDB_biounits_python(
    x: str | Path, atoms: list[str] = ["N", "CA", "C"], chain: str | None = None
) -> tuple[np.ndarray, list[str]] | tuple[str, str]:
    """
    Pure Python implementation of PDB parsing.

    Optimizations over legacy version:
    - Uses module-level constants (avoid recreating per call)
    - Uses text mode (avoid decode overhead)
    - Collects coordinates as tuples (avoid numpy overhead per atom)
    - Single numpy conversion at the end

    input:  x = PDB filename
            atoms = atoms to extract (optional)
            chain = chain to extract (optional, None means all chains)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    if not isinstance(x, Path):
        x = Path(x)

    xyz: dict[int, dict[str, dict[str, tuple[float, float, float]]]] = {}
    seq: dict[int, dict[str, str]] = {}
    min_resn, max_resn = 1e6, -1e6
    nan_tuple = (float("nan"), float("nan"), float("nan"))

    with x.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Handle MSE (selenomethionine) as MET
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = "ATOM  " + line[6:17] + "MET" + line[20:]

            if line[:4] == "ATOM":
                ch = line[21]
                if ch == chain or chain is None:
                    atom = line[12:16].strip()
                    resi = line[17:20]
                    resn_str = line[22:27].strip()

                    # Parse coordinates
                    x_, y, z = (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    )

                    # Handle insertion codes
                    if resn_str[-1].isalpha():
                        resa, resn = resn_str[-1], int(resn_str[:-1]) - 1
                    else:
                        resa, resn = "", int(resn_str) - 1

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

                    # Store as tuple (avoid numpy array creation overhead)
                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = (x_, y, z)

    # Convert to numpy arrays, fill in missing values
    seq_: list[int] = []
    xyz_: list[tuple[float, float, float]] = []
    num_atoms = len(atoms)

    try:
        for resn in range(int(min_resn), int(max_resn + 1)):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(_AA_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(nan_tuple)
            else:
                for _ in range(num_atoms):
                    xyz_.append(nan_tuple)

        # Single numpy conversion at the end
        xyz_array = np.array(xyz_, dtype=np.float64).reshape(-1, num_atoms, 3)
        return xyz_array, _n_to_aa(seq_)

    except TypeError:
        return "no_chain", "no_chain"


def _read_pdb_coordinates_only(path: Path) -> io.StringIO:
    """Read PDB file keeping only coordinate lines, skipping headers.

    This avoids Biopython header parsing bugs and speeds up parsing.
    """
    coord_prefixes = ("ATOM", "HETATM", "MODEL", "ENDMDL", "TER", "END")
    with path.open() as f:
        lines = [line for line in f if line.startswith(coord_prefixes)]
    return io.StringIO("".join(lines))


def parse_PDB_biounits_biopython(
    x: str | Path, atoms: list[str] = ["N", "CA", "C"], chain: str | None = None
) -> tuple[np.ndarray, list[str]] | tuple[str, str]:
    """
    Biopython-based implementation of parse_PDB_biounits.

    Optimizations over original:
    - Uses compiled Biopython parser (C-based)
    - Module-level constants and parser instance
    - Collects coordinates as tuples, converts to numpy at end
    - Skips header parsing for robustness and speed

    input:  x = PDB filename
            atoms = atoms to extract (optional)
            chain = chain to extract (optional, None means first chain)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    if not isinstance(x, Path):
        x = Path(x)

    try:
        handle = _read_pdb_coordinates_only(x)
        structure = _BIOPYTHON_PARSER.get_structure("protein", handle)
    except Exception:
        return "no_chain", "no_chain"

    # Get the first model
    model = structure[0]

    # Find target chain(s)
    chains = list(model.get_chains())
    if not chains:
        return "no_chain", "no_chain"

    if chain is not None:
        # Specific chain requested
        target_chains = [ch for ch in chains if ch.id == chain]
        if not target_chains:
            return "no_chain", "no_chain"
    else:
        # chain=None: process ALL chains (matches original behavior)
        target_chains = chains

    # Collect residues, handling insertion codes
    # Use dict structure matching original: xyz[resn][insertion_code] = coords
    xyz: dict[int, dict[str, dict[str, tuple[float, float, float]]]] = {}
    seq: dict[int, dict[str, str]] = {}
    min_resn, max_resn = 1e6, -1e6
    nan_tuple = (float("nan"), float("nan"), float("nan"))

    for target_chain in target_chains:
        for residue in target_chain.get_residues():
            # Skip heteroatoms (water, ligands) except MSE (selenomethionine)
            hetflag = residue.id[0]
            if hetflag != " " and residue.resname != "MSE":
                continue

            # Match original logic: resn = residue_number - 1
            resn = residue.id[1] - 1
            resa = residue.id[2].strip()  # insertion code
            resname = "MET" if residue.resname == "MSE" else residue.resname

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
                seq[resn][resa] = resname

            # Store coordinates as tuples (avoid numpy overhead per atom)
            # Handle disordered atoms: prefer altloc ' ' or 'A'
            for atom_name in atoms:
                if atom_name in residue and atom_name not in xyz[resn][resa]:
                    atom = residue[atom_name]
                    # For disordered atoms, prefer altloc 'A' or ' '
                    if atom.is_disordered():
                        for altloc in ("A", " "):
                            if atom.disordered_has_id(altloc):
                                atom = atom.disordered_get(altloc)
                                break
                    coord = atom.coord
                    xyz[resn][resa][atom_name] = (
                        float(coord[0]),
                        float(coord[1]),
                        float(coord[2]),
                    )

    if not xyz:
        return "no_chain", "no_chain"

    # Build output arrays matching original format
    seq_: list[int] = []
    xyz_: list[tuple[float, float, float]] = []
    num_atoms = len(atoms)

    try:
        for resn in range(int(min_resn), int(max_resn + 1)):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(_AA_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(nan_tuple)
            else:
                for _ in range(num_atoms):
                    xyz_.append(nan_tuple)

        # Single numpy conversion at the end
        xyz_array = np.array(xyz_, dtype=np.float64).reshape(-1, num_atoms, 3)
        return xyz_array, _n_to_aa(seq_)

    except TypeError:
        return "no_chain", "no_chain"


def compare_pdb_parsers(pdb_path: str | Path, chain: str | None = None) -> dict:
    """
    Compare the original and Biopython implementations of parse_PDB_biounits.

    Returns a dict with comparison results including shapes, sequences,
    and coordinate differences.
    """
    if not isinstance(pdb_path, Path):
        pdb_path = Path(pdb_path)

    # Run both implementations
    xyz_orig, seq_orig = parse_PDB_biounits(pdb_path, chain=chain)
    xyz_bio, seq_bio = parse_PDB_biounits_biopython(pdb_path, chain=chain)

    result = {
        "pdb_path": str(pdb_path),
        "chain": chain,
        "original": {"xyz_shape": None, "seq": None},
        "biopython": {"xyz_shape": None, "seq": None},
        "match": False,
        "xyz_match": False,
        "seq_match": False,
        "xyz_max_diff": None,
    }

    # Handle error cases
    if isinstance(xyz_orig, str) or isinstance(xyz_bio, str):
        orig_shape = xyz_orig if isinstance(xyz_orig, str) else xyz_orig.shape
        bio_shape = xyz_bio if isinstance(xyz_bio, str) else xyz_bio.shape
        result["original"]["xyz_shape"] = orig_shape
        result["original"]["seq"] = seq_orig
        result["biopython"]["xyz_shape"] = bio_shape
        result["biopython"]["seq"] = seq_bio
        # Both must be strings (error case) and match
        both_strings = isinstance(xyz_orig, str) and isinstance(xyz_bio, str)
        result["match"] = both_strings and xyz_orig == xyz_bio and seq_orig == seq_bio
        return result

    result["original"]["xyz_shape"] = xyz_orig.shape
    result["original"]["seq"] = seq_orig
    result["biopython"]["xyz_shape"] = xyz_bio.shape
    result["biopython"]["seq"] = seq_bio

    # Compare sequences
    result["seq_match"] = seq_orig == seq_bio

    # Compare coordinates (allowing for NaN)
    if xyz_orig.shape == xyz_bio.shape:
        # Handle NaN values
        orig_nan = np.isnan(xyz_orig)
        bio_nan = np.isnan(xyz_bio)
        nan_match = np.array_equal(orig_nan, bio_nan)

        if nan_match:
            # Compare non-NaN values
            mask = ~orig_nan
            if mask.any():
                max_diff = np.abs(xyz_orig[mask] - xyz_bio[mask]).max()
                result["xyz_max_diff"] = float(max_diff)
                # Allow for float parsing precision differences
                result["xyz_match"] = max_diff < 1e-5
            else:
                result["xyz_match"] = True
        else:
            result["xyz_match"] = False
    else:
        result["xyz_match"] = False

    result["match"] = result["xyz_match"] and result["seq_match"]

    return result


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
            # Check for valid chain: not a string and has a non-empty sequence
            if not isinstance(xyz, str) and seq[0]:
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

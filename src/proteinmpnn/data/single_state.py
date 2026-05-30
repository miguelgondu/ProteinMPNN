"""Single-state protein design input formatter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from Bio.PDB import PDBParser

from proteinmpnn.data.config import DesignableResidue, SingleStateConfig
from proteinmpnn.utils.constants import AA3_TO_AA
from proteinmpnn.utils.pdb import get_neighbors_within_radius
from proteinmpnn.utils.residue import (
    parse_residue,
    parse_residue_list,
    parse_residue_range,
    validate_symmetric_groups,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class SingleStateDesignInput:
    """Formatter for single-PDB protein design input.

    This class handles the preparation of design specifications for single-state
    protein design with ProteinMPNN. It parses PDB structures, identifies designable
    residues, and handles symmetry constraints.

    Args:
        pdb_path: Path to the input PDB file.
        designable_res: Comma-separated residue specifications
            (e.g., "A10,A12-A15").
        default_design_setting: Default amino acids allowed for mutation
            ("all" or specific AAs).
        symmetric_res: Comma-separated symmetry constraints
            (e.g., "A10:B10,A11:B11").
        cluster_center: Center residue(s) for cluster-based design
            (e.g., "A10" or "A10,A15").
        cluster_radius: Radius in Angstroms for cluster-based selection.

    Example:
        >>> design = SingleStateDesignInput(
        ...     pdb_path="structure.pdb",
        ...     designable_res="A10-A20",
        ...     symmetric_res="A10:B10"
        ... )
        >>> design.generate_json("output.json")
    """

    def __init__(
        self,
        pdb_path: str | Path,
        designable_res: str = "",
        default_design_setting: str = "all",
        symmetric_res: str = "",
        cluster_center: str = "",
        cluster_radius: float = 10.0,
    ) -> None:
        self.pdb_path = Path(pdb_path)
        self._validate_pdb_path()

        self.parser = PDBParser(QUIET=True)
        self.default_design_setting = default_design_setting

        # Parse structure first to set up internal state
        self._pdbids: dict[str, tuple[str, str, int]] = {}
        self._chain_seqs: dict[str, str] = {}
        self._process_structure()

        # Parse designable residues
        self._design_res: list[tuple[str, int]] = []
        if designable_res:
            self._design_res = parse_residue_list(designable_res)

        # Parse cluster center and add neighbors
        if cluster_center:
            cluster_mut = self._parse_cluster_center(cluster_center, cluster_radius)
            self._design_res.extend(cluster_mut)
            # Remove duplicates while preserving order
            seen: set[tuple[str, int]] = set()
            unique_res: list[tuple[str, int]] = []
            for res in self._design_res:
                if res not in seen:
                    seen.add(res)
                    unique_res.append(res)
            self._design_res = unique_res

        # Parse symmetric residues
        self._symmetric_res: list[dict[str, list[tuple[str, int]]]] = []
        if symmetric_res:
            self._symmetric_res = self._parse_symmetric_res(symmetric_res)

        # Update design residues based on symmetry constraints
        if cluster_center and self._symmetric_res:
            self._update_design_res_from_symmetry(cluster_mut)

    def _validate_pdb_path(self) -> None:
        """Validate the PDB file exists and has correct extension."""
        if not self.pdb_path.exists():
            raise ValueError(f"PDB file does not exist: {self.pdb_path}")
        if self.pdb_path.suffix.lower() != ".pdb":
            raise ValueError(f"Expected .pdb file: {self.pdb_path}")

    def _process_structure(self) -> None:
        """Parse PDB and build internal state (sequences and residue mappings)."""
        structure = self.parser.get_structure(self.pdb_path.stem, self.pdb_path)

        for chain in structure.get_chains():
            self._chain_seqs[chain.id] = []
            res_index_chain = 1
            residues = list(chain.get_residues())

            for residue in residues:
                num_id = residue.id[1]
                pdbid = chain.id + str(num_id)

                # Add gapped residues
                if residue != residues[0]:
                    n_gaps = 0
                    while True:
                        prev_res = chain.id + str(num_id - n_gaps - 1)
                        if prev_res not in self._pdbids:
                            n_gaps += 1
                        else:
                            break

                    for i in range(n_gaps):
                        prev_res = chain.id + str(num_id - n_gaps + i)
                        self._pdbids[prev_res] = (
                            AA3_TO_AA["XXX"],
                            chain.id,
                            res_index_chain,
                        )
                        self._chain_seqs[chain.id].append(AA3_TO_AA["XXX"])
                        res_index_chain += 1

                # Add current residue
                self._pdbids[pdbid] = (
                    AA3_TO_AA.get(residue.get_resname(), "X"),
                    chain.id,
                    res_index_chain,
                )
                self._chain_seqs[chain.id].append(
                    AA3_TO_AA.get(residue.get_resname(), "X")
                )
                res_index_chain += 1

            # Join sequence into string
            self._chain_seqs[chain.id] = "".join(
                x for x in self._chain_seqs[chain.id] if x is not None
            )

    def _parse_cluster_center(
        self, cluster_center: str, cluster_radius: float
    ) -> list[tuple[str, int]]:
        """Parse cluster center specification and find neighboring residues."""
        centers: list[tuple[str, int]] = []
        items = [s for s in cluster_center.strip().split(",") if s]

        for item in items:
            if "-" in item:
                centers.extend(parse_residue_range(item))
            else:
                centers.append(parse_residue(item))

        design_res: list[tuple[str, int]] = []
        for center in centers:
            neighbors = get_neighbors_within_radius(
                self.pdb_path, center, cluster_radius
            )
            design_res.extend(neighbors)

        # Remove duplicates
        return list(set(design_res))

    def _parse_symmetric_res(
        self, symmetric_str: str
    ) -> list[dict[str, list[tuple[str, int]]]]:
        """Parse symmetry constraints string."""
        items = [s for s in symmetric_str.strip().split(",") if s]

        symmetric_res: list[dict[str, list[tuple[str, int]]]] = []
        for item in items:
            if ":" not in item:
                raise ValueError(f"No colon detected in symmetric res: {item}.")

            symmetry_dict = self._check_symmetry_validity(item)
            symmetric_res.append(symmetry_dict)

        return symmetric_res

    def _check_symmetry_validity(
        self, symmetric_item: str
    ) -> dict[str, list[tuple[str, int]]]:
        """Validate and parse a single symmetry constraint item."""
        split_item = symmetric_item.split(":")

        symmetry_dict: dict[str, list[tuple[str, int]]] = {}
        for subitem in split_item:
            if "-" in subitem:
                res_range = parse_residue_range(subitem)
                symmetry_dict[subitem] = res_range
            else:
                res_ch, res_idx = parse_residue(subitem)
                symmetry_dict[subitem] = [(res_ch, res_idx)]

        # Validate all groups have same length
        groups = list(symmetry_dict.values())
        validate_symmetric_groups(groups)

        return symmetry_dict

    def _update_design_res_from_symmetry(
        self, cluster_mutations: Sequence[tuple[str, int]]
    ) -> None:
        """Add symmetric partners of cluster mutations to design residues."""
        for symmetry in self._symmetric_res:
            items = list(symmetry.values())
            for cm in cluster_mutations:
                for chain_residues in items:
                    if cm in chain_residues:
                        cm_idx = chain_residues.index(cm)
                        for add_item in items:
                            if cm_idx < len(add_item):
                                mut_to_add = add_item[cm_idx]
                                if (
                                    mut_to_add not in self._design_res
                                    and mut_to_add[1] == cm[1]
                                ):
                                    self._design_res.append(mut_to_add)

        # Sort design residues
        self._design_res = sorted(
            sorted(self._design_res, key=lambda x: x[1]), key=lambda x: x[0]
        )

    def _build_mutable_list(self) -> list[dict[str, str | int]]:
        """Build list of mutable residue specifications."""
        mutable: list[dict[str, str | int]] = []
        for resind in self._design_res:
            res_id = resind[0] + str(resind[1])
            if res_id in self._pdbids and self._pdbids[res_id][0] != "X":
                mutable.append(
                    {
                        "chain": self._pdbids[res_id][1],
                        "resid": self._pdbids[res_id][2],
                        "WTAA": self._pdbids[res_id][0],
                        "MutTo": self.default_design_setting,
                    }
                )
        return mutable

    def _build_symmetric_list(self) -> list[list[str]]:
        """Build list of symmetric position groups."""
        symmetric: list[list[str]] = []
        for symmetry in self._symmetric_res:
            values = list(symmetry.values())

            for tied_pos in zip(*values):
                skip_tie = False
                sym_res: list[str] = []
                for pos in tied_pos:
                    res_id = pos[0] + str(pos[1])
                    if res_id not in self._pdbids or self._pdbids[res_id][0] == "X":
                        skip_tie = True
                        break
                    sym_res.append(
                        self._pdbids[res_id][1] + str(self._pdbids[res_id][2])
                    )

                if not skip_tie:
                    symmetric.append(sym_res)

        return symmetric

    def to_config(self) -> SingleStateConfig:
        """Return the design specification as a validated Pydantic model.

        Returns:
            A SingleStateConfig object containing the design specification.
        """
        mutable = self._build_mutable_list()
        symmetric = self._build_symmetric_list()

        return SingleStateConfig(
            sequence=self._chain_seqs,
            designable=[
                DesignableResidue(
                    chain=r["chain"],
                    resid=r["resid"],
                    WTAA=r["WTAA"],
                    MutTo=r["MutTo"],
                )
                for r in mutable
            ],
            symmetric=symmetric,
        )

    def generate_json(self, out_path: str | Path) -> None:
        """Generate JSON config file for ProteinMPNN.

        Args:
            out_path: Output file path for the JSON configuration.
        """
        self.to_config().to_json(out_path)

"""Multi-state protein design input formatter."""

from __future__ import annotations

import itertools
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy as np
from Bio.PDB import PDBIO, PDBParser

from proteinmpnn.data.config import DesignableResidue, MultiStateConfig
from proteinmpnn.utils.constants import AA3_TO_AA, CHAIN_IDS
from proteinmpnn.utils.pdb import (
    NotDisordered,
    calculate_min_inter_state_distance,
    check_structure_bounds,
    get_neighbors_within_radius,
)
from proteinmpnn.utils.residue import (
    parse_residue,
    parse_residue_range,
    validate_symmetric_groups,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class MultiStateDesignInput:
    """Formatter for multi-state protein design input.

    This class handles the preparation of design specifications for multi-state
    protein design (MSD) with ProteinMPNN. It combines multiple PDB structures,
    manages chain remapping, and handles symmetry constraints between states.

    Args:
        pdb_dir: Directory containing PDB files (at least 2 required).
        constraints: Required symmetry constraints between states
            (e.g., "PDB1:A10:1.0,PDB2:A10:0.5;PDB1:A11:1.0,PDB2:A11:0.5").
        designable_res: Residues to design, with PDB prefix
            (e.g., "PDB1:A10,A12-A15;PDB2:B5-B10").
        default_design_setting: Default amino acids allowed for mutation.
        cluster_center: Center residue(s) for cluster-based design with PDB prefix.
        cluster_radius: Radius in Angstroms for cluster-based residue selection.
        gap: Distance in Angstroms to separate states in combined structure.
        validation_tries: Number of attempts to resolve clashes (0 to skip validation).
        bidirectional: If True, apply bidirectional symmetry (for dimeric constraints).

    Example:
        >>> design = MultiStateDesignInput(
        ...     pdb_dir="structures/",
        ...     constraints="state1:A10:1.0,state2:A10:0.5"
        ... )
        >>> design.generate_json("output.json")
    """

    def __init__(
        self,
        pdb_dir: str | Path,
        constraints: str,
        designable_res: str = "",
        default_design_setting: str = "all",
        cluster_center: str = "",
        cluster_radius: float = 10.0,
        gap: float = 1000.0,
        validation_tries: int = 0,
        bidirectional: bool = False,
    ) -> None:
        self.pdb_dir = Path(pdb_dir)
        self._validate_pdb_dir()

        self.parser = PDBParser(QUIET=True)
        self.default_design_setting = default_design_setting
        self.bidirectional = bidirectional

        # Combine PDFs into one structure with remapped chains
        self.chain_dict = self._combine_pdbs(gap, validation_tries)

        # Internal state
        self._pdbids: dict[str, tuple[str, str, int]] = {}
        self._chain_seqs: dict[str, str] = {}
        self._beta_dict: dict[str, float] = {}

        # Parse designable residues (needs chain_dict from _combine_pdbs)
        self._design_res: list[tuple[str, int]] = []
        if designable_res:
            self._design_res = self._parse_designable_res(designable_res)

        # Parse cluster center and add neighbors
        if cluster_center:
            cluster_mut = self._parse_cluster_center(cluster_center, cluster_radius)
            self._design_res.extend(cluster_mut)
            # Remove duplicates
            self._design_res = list(set(self._design_res))

        # Parse constraints (also populates beta_dict)
        self._symmetric_res: list[dict[str, list[tuple[str, int]]]] = []
        self._symmetric_res = self._parse_constraints(constraints)

        # Update design residues based on symmetry constraints
        if cluster_center and self._symmetric_res:
            self._update_design_res_from_symmetry(cluster_mut)

        # Apply bidirectional symmetry if requested
        if self.bidirectional:
            self._apply_bidirectional()

        # Process structure to build sequences
        self._process_structure()

    def _validate_pdb_dir(self) -> None:
        """Validate the PDB directory exists and contains enough files."""
        if not self.pdb_dir.is_dir():
            raise ValueError(f"Directory does not exist: {self.pdb_dir}")

        self.pdb_list = sorted(self.pdb_dir.glob("*.pdb"))
        if len(self.pdb_list) < 2:
            raise ValueError(
                f"Multi-state requires at least 2 PDB files, found {len(self.pdb_list)}"
            )

    def _combine_pdbs(
        self, gap: float, validation_tries: int
    ) -> dict[str, dict[str, str]]:
        """Combine multiple PDBs into one structure with remapped chains.

        Args:
            gap: Distance in Angstroms to separate states.
            validation_tries: Number of attempts to resolve clashes (0 to skip).

        Returns:
            Mapping from PDB name -> original chain -> remapped chain.
        """
        min_dist = 0.0
        tries = 0
        # Sort combos by sum of increments to reduce spread
        combos = sorted(
            itertools.product([0, 1, 2, 3, 4, 5], repeat=3), key=lambda x: (sum(x), x)
        )
        rand = Random(0)

        while min_dist < 100.0:
            initial_pdb = self.pdb_list[0]
            target = self.parser.get_structure("main", initial_pdb)

            # Initialize chain dict with first PDB
            init_ch = [c.id for c in target.get_chains()]
            chain_dict: dict[str, dict[str, str]] = {
                initial_pdb.stem: {ch: ch for ch in init_ch}
            }
            no_duplicates = list(init_ch)

            io = PDBIO()
            chain_inc = 0

            for model in target:
                for inc, pdb in enumerate(self.pdb_list[1:]):
                    mobile = self.parser.get_structure("mobile", pdb)
                    mobile_dict: dict[str, str] = {}

                    for m in mobile:
                        chain_list = list(m)
                        for chain in chain_list:
                            m.detach_child(chain.id)

                        for chain in chain_list:
                            # Rename chains to avoid conflicts
                            original_id = chain.id
                            tmp = original_id
                            while tmp in no_duplicates:
                                chain_inc += 1
                                tmp = CHAIN_IDS[chain_inc]
                            chain.id = tmp
                            mobile_dict[original_id] = chain.id
                            no_duplicates.append(chain.id)

                            # Add chain to target structure
                            model.add(chain)

                            # Increment position to separate from other states
                            inc_3d = np.array(combos[inc + 1]) * gap
                            for residue in chain:
                                for atom in residue:
                                    if atom.is_disordered():
                                        try:
                                            atom.disordered_select("A")
                                        except KeyError as e:
                                            raise ValueError(
                                                "Failed to resolve disordered residues"
                                            ) from e
                                    atom.set_coord(atom.get_coord() + inc_3d)

                    chain_dict[pdb.stem] = mobile_dict

            # Skip validation if disabled
            if validation_tries == 0:
                print("Skipping validation - Multi-state integration complete!")
                break

            # Check for clashes between states
            min_dist = calculate_min_inter_state_distance(target, chain_dict)

            if min_dist < 100.0:
                rand.shuffle(combos)
                tries += 1
                if tries >= validation_tries:
                    raise RuntimeError(
                        "Multi-state integration failed: clashes between states."
                    )
                print("Multi-state integration failed (clashes) - retrying...")

        # Save combined PDB
        msd_dir = self.pdb_dir / "msd"
        msd_dir.mkdir(exist_ok=True)
        msd_pdb = msd_dir / "msd.pdb"

        if validation_tries > 0:
            check_structure_bounds(target)
            print(f"Multi-state integration validated! Saving at: {msd_pdb}")

        io.set_structure(target)
        io.save(str(msd_pdb), select=NotDisordered())

        # Update paths to point to combined structure
        self.pdb_dir = msd_dir
        self.pdb_list = [msd_pdb]

        return chain_dict

    def _process_structure(self) -> None:
        """Parse combined PDB and build internal state."""
        structure = self.parser.get_structure("msd", self.pdb_list[0])

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

            self._chain_seqs[chain.id] = "".join(
                x for x in self._chain_seqs[chain.id] if x is not None
            )

    def _parse_designable_res(self, design_str: str) -> list[tuple[str, int]]:
        """Parse designable residues with PDB prefixes."""
        design_per_state = [d for d in design_str.strip().split(";") if d]
        design_res: list[tuple[str, int]] = []

        for dps in design_per_state:
            parts = [p for p in dps.strip().split(":") if p]
            if len(parts) != 2:
                raise ValueError(f"Invalid designable_res format: {dps}")
            pdb_name, des_info = parts

            items = [s for s in des_info.strip().split(",") if s]
            for item in items:
                if "-" in item:
                    range_res = parse_residue_range(item)
                    # Remap chains
                    for chain, idx in range_res:
                        new_chain = self.chain_dict[pdb_name][chain]
                        design_res.append((new_chain, idx))
                else:
                    item_ch, item_idx = parse_residue(item)
                    new_chain = self.chain_dict[pdb_name][item_ch]
                    design_res.append((new_chain, item_idx))

        return design_res

    def _parse_cluster_center(
        self, cluster_center: str, cluster_radius: float
    ) -> list[tuple[str, int]]:
        """Parse cluster center specification for multi-state."""
        cluster_per_state = [c for c in cluster_center.strip().split(";") if c]
        centers: list[tuple[str, int]] = []

        for cps in cluster_per_state:
            parts = [c for c in cps.strip().split(":") if c]
            if len(parts) != 2:
                raise ValueError(f"Invalid cluster_center format: {cps}")
            pdb_name, clus_info = parts

            items = [s for s in clus_info.strip().split(",") if s]
            for item in items:
                if "-" in item:
                    range_res = parse_residue_range(item)
                    for chain, idx in range_res:
                        new_chain = self.chain_dict[pdb_name][chain]
                        centers.append((new_chain, idx))
                else:
                    item_ch, item_idx = parse_residue(item)
                    new_chain = self.chain_dict[pdb_name][item_ch]
                    centers.append((new_chain, item_idx))

        design_res: list[tuple[str, int]] = []
        for center in centers:
            neighbors = get_neighbors_within_radius(
                self.pdb_list[0], center, cluster_radius
            )
            design_res.extend(neighbors)

        return list(set(design_res))

    def _parse_constraints(
        self, constraints_str: str
    ) -> list[dict[str, list[tuple[str, int]]]]:
        """Parse MSD constraints string and populate beta_dict."""
        symm_per_constraint = [d for d in constraints_str.strip().split(";") if d]

        symmetric_res: list[dict[str, list[tuple[str, int]]]] = []
        for spc in symm_per_constraint:
            items = [s for s in spc.strip().split(",") if s]

            adj_symm_items: list[str] = []
            pdb_names: list[str] = []

            for item in items:
                parts = [si for si in item.strip().split(":") if si]
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid constraint format: {item}. "
                        "Expected 'PDB_NAME:RESIDUE:BETA'"
                    )
                pdb_name, symm_res, beta = parts

                adj_symm_items.append(symm_res)
                adj_chain = self.chain_dict[pdb_name][symm_res[0]]
                self._beta_dict[adj_chain] = float(beta)
                pdb_names.append(pdb_name)

            item_str = ":".join(adj_symm_items)
            if ":" not in item_str:
                raise ValueError(f"No colon detected in symmetric res: {item_str}.")

            symmetry_dict = self._check_symmetry_validity(item_str, pdb_names)
            symmetric_res.append(symmetry_dict)

        return symmetric_res

    def _check_symmetry_validity(
        self, symmetric_item: str, pdb_names: Sequence[str]
    ) -> dict[str, list[tuple[str, int]]]:
        """Validate and parse a symmetry constraint item."""
        split_item = symmetric_item.split(":")

        symmetry_dict: dict[str, list[tuple[str, int]]] = {}
        for subitem, pdb_name in zip(split_item, pdb_names):
            # Replace chain label with remapped chain
            adj_subitem = subitem.replace(
                subitem[0], self.chain_dict[pdb_name][subitem[0]]
            )

            if "-" in subitem:
                res_range = self._parse_range_with_remap(subitem, pdb_name)
                symmetry_dict[adj_subitem] = res_range
            else:
                res_ch, res_idx = parse_residue(subitem)
                new_chain = self.chain_dict[pdb_name][res_ch]
                symmetry_dict[adj_subitem] = [(new_chain, res_idx)]

        groups = list(symmetry_dict.values())
        validate_symmetric_groups(groups)

        return symmetry_dict

    def _parse_range_with_remap(
        self, range_item: str, pdb_name: str
    ) -> list[tuple[str, int]]:
        """Parse a residue range and remap chains for multi-state."""
        base_range = parse_residue_range(range_item)
        return [(self.chain_dict[pdb_name][ch], idx) for ch, idx in base_range]

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

        self._design_res = sorted(
            sorted(self._design_res, key=lambda x: x[1]), key=lambda x: x[0]
        )

    def _apply_bidirectional(self) -> None:
        """Apply bidirectional symmetry for dimeric constraints."""
        for sr in self._symmetric_res:
            items = list(sr.values())
            if len(items) == 2:
                # Flip one position list for opposite direction design
                sr[list(sr.keys())[1]] = items[1][::-1]
            else:
                raise ValueError(
                    f"Bidirectional only supports 2 subunits, got {len(items)}."
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

    def to_config(self) -> MultiStateConfig:
        """Return the design specification as a validated Pydantic model.

        Returns:
            A MultiStateConfig object containing the design specification.
        """
        mutable = self._build_mutable_list()
        symmetric = self._build_symmetric_list()

        return MultiStateConfig(
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
            tied_betas=self._beta_dict,
            chain_key=self.chain_dict,
        )

    def generate_json(self, out_path: str | Path) -> None:
        """Generate JSON config file for ProteinMPNN.

        Args:
            out_path: Output file path for the JSON configuration.
        """
        self.to_config().to_json(out_path)

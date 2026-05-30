"""Factory function and backward compatibility for protein design input formatters.

This module provides:
- `create_design_input()`: Factory function to create the appropriate formatter
- `ProteinDesignInputFormatter`: Backward compatibility alias

For new code, prefer using the specialized classes directly:
- `SingleStateDesignInput` for single-PDB design
- `MultiStateDesignInput` for multi-state design
"""

from __future__ import annotations

from pathlib import Path

from proteinmpnn.data.multi_state import MultiStateDesignInput
from proteinmpnn.data.single_state import SingleStateDesignInput


def create_design_input(
    pdb_dir: str | Path,
    designable_res: str = "",
    default_design_setting: str = "all",
    symmetric_res: str = "",
    cluster_center: str = "",
    cluster_radius: float = 10.0,
    gap: float = 1000.0,
    validation_tries: int = 0,
    bidirectional: bool = False,
    constraints: str = "",
    multi_state: bool = False,
) -> SingleStateDesignInput | MultiStateDesignInput:
    """Factory function to create the appropriate design input formatter.

    Maintains backward compatibility with original ProteinDesignInputFormatter API.
    For new code, prefer using SingleStateDesignInput or MultiStateDesignInput directly.

    Args:
        pdb_dir: Directory containing PDB file(s).
        designable_res: Residues to design (e.g., "A10,A12-A15" for
            single-state, "PDB1:A10;PDB2:B10" for multi-state).
        default_design_setting: Default amino acids for mutation
            ("all" or specific AAs).
        symmetric_res: Symmetry constraints for single-state (e.g., "A10:B10").
        cluster_center: Center residue(s) for cluster-based design.
        cluster_radius: Radius in Angstroms for cluster-based selection.
        gap: Distance to separate states in multi-state design.
        validation_tries: Attempts to resolve clashes in multi-state (0=skip).
        bidirectional: Apply bidirectional symmetry (multi-state only).
        constraints: Required constraints for multi-state design.
        multi_state: If True, use multi-state design mode.

    Returns:
        Either a SingleStateDesignInput or MultiStateDesignInput instance.

    Raises:
        ValueError: If arguments are incompatible or required arguments are missing.

    Examples:
        Single-state design:
        >>> design = create_design_input(
        ...     pdb_dir="structures/",
        ...     designable_res="A10-A20"
        ... )

        Multi-state design:
        >>> design = create_design_input(
        ...     pdb_dir="structures/",
        ...     multi_state=True,
        ...     constraints="state1:A10:1.0,state2:A10:0.5"
        ... )
    """
    pdb_dir = Path(pdb_dir)

    # Validate option compatibility
    if bidirectional and not multi_state:
        raise ValueError("Cannot enable bidirectional without enabling multi_state.")
    if constraints and not multi_state:
        raise ValueError("Cannot specify constraints without enabling multi_state.")
    if multi_state and not constraints:
        raise ValueError("Cannot enable multi_state without specifying constraints.")
    if constraints and symmetric_res:
        raise ValueError("Cannot specify both constraints and symmetric_res.")

    if multi_state:
        return MultiStateDesignInput(
            pdb_dir=pdb_dir,
            constraints=constraints,
            designable_res=designable_res,
            default_design_setting=default_design_setting,
            cluster_center=cluster_center,
            cluster_radius=cluster_radius,
            gap=gap,
            validation_tries=validation_tries,
            bidirectional=bidirectional,
        )

    # Single-state mode requires exactly 1 PDB file
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if len(pdb_files) == 0:
        raise ValueError(f"No PDB files found in {pdb_dir}")
    if len(pdb_files) > 1:
        raise ValueError(
            f"Single-state requires 1 PDB file, found {len(pdb_files)}. "
            "Use multi_state=True for multiple files."
        )

    return SingleStateDesignInput(
        pdb_path=pdb_files[0],
        designable_res=designable_res,
        default_design_setting=default_design_setting,
        symmetric_res=symmetric_res,
        cluster_center=cluster_center,
        cluster_radius=cluster_radius,
    )


# Backward compatibility alias
ProteinDesignInputFormatter = create_design_input

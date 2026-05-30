"""Pydantic models for ProteinMPNN design output configurations."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class DesignableResidue(BaseModel):
    """A single residue that can be mutated."""

    chain: str = Field(description="Chain identifier")
    resid: int = Field(description="Residue index (1-based, sequential within chain)")
    WTAA: str = Field(description="Wild-type amino acid (1-letter code)")
    MutTo: str = Field(default="all", description="Allowed amino acids for mutation")


class SingleStateConfig(BaseModel):
    """Output config for single-state protein design."""

    sequence: dict[str, str] = Field(
        default_factory=dict, description="Chain ID -> amino acid sequence"
    )
    designable: list[DesignableResidue] = Field(
        default_factory=list, description="List of designable residues"
    )
    symmetric: list[list[str]] = Field(
        default_factory=list, description="List of tied position groups"
    )

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Write config to JSON file.

        Args:
            path: Output file path.
            indent: JSON indentation level.
        """
        Path(path).write_text(self.model_dump_json(indent=indent))


class MultiStateConfig(SingleStateConfig):
    """Output config for multi-state protein design.

    Extends SingleStateConfig with additional MSD-specific fields.
    """

    tied_betas: dict[str, float] = Field(
        default_factory=dict, description="Chain ID -> beta weight for MSD"
    )
    chain_key: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="PDB name -> original chain -> remapped chain",
    )

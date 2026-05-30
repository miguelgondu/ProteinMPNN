"""Result dataclasses for ProteinMPNN inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SequenceResult:
    """A single designed sequence result.

    Attributes:
        sequence: The designed amino acid sequence (chains separated by '/').
        score: The negative log probability score (lower is better).
        seq_recovery: Fraction of residues matching the native sequence.
        temperature: Sampling temperature used for this sequence.
        sample_index: Index of this sample within the batch.
    """

    sequence: str
    score: float
    seq_recovery: float
    temperature: float
    sample_index: int

    def to_fasta_header(self) -> str:
        """Generate a FASTA header line for this sequence."""
        return (
            f">T={self.temperature}, sample={self.sample_index}, "
            f"score={self.score:.4f}, seq_recovery={self.seq_recovery:.4f}"
        )


@dataclass
class NativeSequence:
    """Native (wild-type) sequence information.

    Attributes:
        name: Protein name from the PDB file.
        sequence: The native amino acid sequence (chains separated by '/').
        score: The negative log probability score for the native sequence.
        fixed_chains: List of chain IDs that were fixed (not designed).
        designed_chains: List of chain IDs that were designed.
        model_name: Name of the ProteinMPNN model used.
    """

    name: str
    sequence: str
    score: float
    fixed_chains: list[str]
    designed_chains: list[str]
    model_name: str

    def to_fasta_header(self) -> str:
        """Generate a FASTA header line for the native sequence."""
        return (
            f">{self.name}, score={self.score:.4f}, "
            f"fixed_chains={self.fixed_chains}, "
            f"designed_chains={self.designed_chains}, "
            f"model_name={self.model_name}"
        )


@dataclass
class DesignResult:
    """Complete result from a ProteinMPNN design run.

    Attributes:
        protein_name: Name of the protein (from PDB file).
        native: Native sequence information and scores.
        sequences: List of designed sequence results.
        probs: Optional probability matrix (L x 21) averaged across samples.
    """

    protein_name: str
    native: NativeSequence
    sequences: list[SequenceResult] = field(default_factory=list)
    probs: np.ndarray | None = None

    def to_fasta(self) -> str:
        """Generate FASTA-formatted string for all sequences.

        Returns:
            FASTA-formatted string with native sequence first,
            followed by all designed sequences.
        """
        lines = []

        # Native sequence first
        lines.append(self.native.to_fasta_header())
        lines.append(self.native.sequence)

        # Then all designed sequences
        for seq_result in self.sequences:
            lines.append(seq_result.to_fasta_header())
            lines.append(seq_result.sequence)

        return "\n".join(lines)

    def to_af2_csv(self) -> str:
        """Generate AlphaFold2-compatible CSV format.

        The AF2 format has each chain sequence comma-separated,
        with a comment containing metadata.

        Returns:
            CSV-formatted string compatible with AlphaFold2 batch input.
        """
        lines = []

        # Native sequence
        chains = self.native.sequence.split("/")
        af2_seqs = "," + ",".join(chains)
        # Sanitize comment (remove commas)
        comment = (
            f"{self.native.name} score={self.native.score:.4f} "
            f"fixed_chains={self.native.fixed_chains} "
            f"designed_chains={self.native.designed_chains} "
            f"model_name={self.native.model_name}"
        ).replace(",", "")
        lines.append(f"{af2_seqs} # {comment}")

        # Designed sequences
        for seq_result in self.sequences:
            chains = seq_result.sequence.split("/")
            af2_seqs = "," + ",".join(chains)
            comment = (
                f"T={seq_result.temperature} sample={seq_result.sample_index} "
                f"score={seq_result.score:.4f} "
                f"seq_recovery={seq_result.seq_recovery:.4f}"
            ).replace(",", "")
            lines.append(f"{af2_seqs} # {comment}")

        return "\n".join(lines)

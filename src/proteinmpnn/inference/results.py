"""Result dataclasses for ProteinMPNN inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from proteinmpnn.model.utils import ALPHABET


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


class ResidueInfo(BaseModel):
    """Info for a single residue position."""

    chain: str
    residue_idx: int


class ConditionalProbsResult(BaseModel):
    """Result from conditional probability computation.

    There is exactly one row per *parsed* residue, in parse order, including
    residues the model could not use. ``observed`` says which rows the model
    actually saw: a residue is unobserved when any of its N, CA, C, O coordinates
    is missing, which happens both for the placeholder rows ``parse_PDB`` inserts
    across gaps in author numbering and for real residues with an incomplete
    backbone.

    **Log probabilities on unobserved rows are meaningless** -- the model ran
    with their coordinates zeroed -- so mask with ``observed`` before using them.
    They are passed through rather than replaced by NaN, which matches upstream
    ProteinMPNN's ``--conditional_probs_only`` output: it likewise emits every
    row and ships the mask beside it.

    Attributes:
        protein_name: Name of the protein (from PDB file).
        model_name: Name of the ProteinMPNN model used.
        log_probs: Log probabilities array of shape [L, 21].
        observed: Boolean array of shape [L]; False where the backbone was unusable.
        native_seq: The parsed sequence, length L, with "X" on unobserved rows.
        n_draws: How many decoding orders were drawn and averaged, in log space.
        residue_info: List of ResidueInfo for each position.
        mode: Whether the probabilities are conditional or unconditional.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protein_name: str
    model_name: str
    log_probs: np.ndarray  # Shape [L, 21]
    observed: np.ndarray  # Shape [L], bool
    native_seq: str  # Length L
    n_draws: int = 1
    residue_info: list[ResidueInfo]
    mode: Literal["conditional", "unconditional"]

    def to_csv(self) -> str:
        """Generate CSV-formatted string with log probabilities.

        Returns:
            CSV string with header comment and one row per residue position.
        """
        lines = []

        # Header comment with metadata
        lines.append(
            f"# protein={self.protein_name}, model={self.model_name}, "
            f"mode={self.mode}, n_draws={self.n_draws}"
        )

        # Column headers. `observed` is included because the log probabilities on
        # unobserved rows are meaningless, and a CSV consumer has no other way to tell.
        header = ["chain", "residue_idx", "observed"] + list(ALPHABET)
        lines.append(",".join(header))

        # Data rows
        for i, res_info in enumerate(self.residue_info):
            row_values = [
                res_info.chain,
                str(res_info.residue_idx),
                str(bool(self.observed[i])).lower(),
            ]
            row_values.extend(f"{self.log_probs[i, j]:.6f}" for j in range(21))
            lines.append(",".join(row_values))

        return "\n".join(lines)

    def to_npz_dict(self) -> dict:
        """Generate dictionary for saving as NPZ file.

        Returns:
            Dictionary with log_probs, residue_info, alphabet, and metadata.
        """
        # Create structured array for residue info
        residue_info_arr = np.array(
            [(r.chain, r.residue_idx) for r in self.residue_info],
            dtype=[("chain", "U1"), ("residue_idx", "i4")],
        )

        return {
            "log_probs": self.log_probs,
            "observed": self.observed,
            "native_seq": self.native_seq,
            "residue_info": residue_info_arr,
            "alphabet": ALPHABET,
            "metadata": {
                "protein_name": self.protein_name,
                "model_name": self.model_name,
                "mode": self.mode,
                "n_draws": self.n_draws,
            },
        }

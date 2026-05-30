"""Output formatting utilities for CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from proteinmpnn.inference.results import DesignResult


def write_fasta(result: DesignResult, output_path: Path) -> None:
    """Write design result to a FASTA file.

    Args:
        result: DesignResult from inference.
        output_path: Path to output FASTA file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_fasta() + "\n")


def write_af2_csv(result: DesignResult, output_path: Path) -> None:
    """Write design result to AlphaFold2-compatible CSV.

    Args:
        result: DesignResult from inference.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_af2_csv() + "\n")


def write_probs(result: DesignResult, output_path: Path) -> None:
    """Write probability matrix to NPZ file.

    Args:
        result: DesignResult from inference (must have probs).
        output_path: Path to output NPZ file.

    Raises:
        ValueError: If result has no probability matrix.
    """
    if result.probs is None:
        raise ValueError("DesignResult has no probability matrix")

    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, probs=result.probs)

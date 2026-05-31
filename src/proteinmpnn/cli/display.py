"""Rich display utilities for CLI output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from proteinmpnn.inference.results import ConditionalProbsResult, DesignResult

console = Console()


def display_design_results(result: DesignResult) -> None:
    """Display design results in a rich table."""
    # Create a table for designed sequences
    table = Table(title=f"Designed Sequences for {result.protein_name}")

    table.add_column("#", style="dim", width=4)
    table.add_column("Temperature", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Recovery", justify="right", style="yellow")
    table.add_column("Sequence", style="dim", overflow="fold", max_width=50)

    # Add native sequence first
    table.add_row(
        "WT",
        "-",
        f"{result.native.score:.4f}",
        "1.0000",
        _truncate_seq(result.native.sequence),
        style="bold",
    )

    # Add designed sequences sorted by score
    sorted_seqs = sorted(result.sequences, key=lambda s: s.score)
    for i, seq in enumerate(sorted_seqs, 1):
        is_best = i == 1
        table.add_row(
            str(i),
            f"{seq.temperature:.2f}",
            f"{seq.score:.4f}",
            f"{seq.seq_recovery:.4f}",
            _truncate_seq(seq.sequence),
            style="bold green" if is_best else None,
        )

    console.print()
    console.print(table)

    # Summary panel
    best = sorted_seqs[0]
    worst = sorted_seqs[-1]
    summary = (
        f"[bold]Native score:[/bold] {result.native.score:.4f}\n"
        f"[bold green]Best score:[/bold green] {best.score:.4f} "
        f"(T={best.temperature}, recovery={best.seq_recovery:.2%})\n"
        f"[bold]Worst score:[/bold] {worst.score:.4f}\n"
        f"[bold]Total sequences:[/bold] {len(result.sequences)}"
    )
    console.print(Panel(summary, title="Summary", border_style="blue"))


def display_probs_results(result: ConditionalProbsResult) -> None:
    """Display probability computation results."""
    import numpy as np

    from proteinmpnn.model.utils import ALPHABET

    # Summary panel
    mode_style = "yellow" if result.mode == "unconditional" else "cyan"
    summary = (
        f"[bold]Protein:[/bold] {result.protein_name}\n"
        f"[bold]Model:[/bold] {result.model_name}\n"
        f"[bold]Mode:[/bold] [{mode_style}]{result.mode}[/{mode_style}]\n"
        f"[bold]Residues:[/bold] {len(result.residue_info)}\n"
        f"[bold]Matrix shape:[/bold] {result.log_probs.shape}"
    )
    console.print()
    console.print(Panel(summary, title="Probability Computation", border_style="blue"))

    # Show top predictions for first few residues
    table = Table(title="Top 3 Predictions per Position (first 10 residues)")
    table.add_column("Position", style="dim")
    table.add_column("Chain", style="cyan")
    table.add_column("1st", justify="center")
    table.add_column("P(1st)", justify="right", style="green")
    table.add_column("2nd", justify="center")
    table.add_column("P(2nd)", justify="right")
    table.add_column("3rd", justify="center")
    table.add_column("P(3rd)", justify="right", style="dim")

    # Show first 10 residues
    for i, res_info in enumerate(result.residue_info[:10]):
        probs = np.exp(result.log_probs[i])
        top_indices = np.argsort(probs)[::-1][:3]

        table.add_row(
            str(res_info.residue_idx),
            res_info.chain,
            ALPHABET[top_indices[0]],
            f"{probs[top_indices[0]]:.3f}",
            ALPHABET[top_indices[1]],
            f"{probs[top_indices[1]]:.3f}",
            ALPHABET[top_indices[2]],
            f"{probs[top_indices[2]]:.3f}",
        )

    console.print(table)

    if len(result.residue_info) > 10:
        console.print(
            f"[dim]... and {len(result.residue_info) - 10} more residues[/dim]"
        )


def _truncate_seq(seq: str, max_len: int = 40) -> str:
    """Truncate sequence for display."""
    if len(seq) <= max_len:
        return seq
    return seq[: max_len - 3] + "..."

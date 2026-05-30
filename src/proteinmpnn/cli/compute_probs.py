"""Compute conditional probabilities CLI command."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal

import typer

from proteinmpnn.cli import app
from proteinmpnn.cli.output import write_probs_csv, write_probs_npz
from proteinmpnn.inference import InferenceRunner
from proteinmpnn.utils.logging import get_logger

logger = get_logger("cli.compute_probs")


@app.command()
def compute_probs(
    pdb_path: Annotated[Path, typer.Argument(help="Path to the input PDB file")],
    model_name: Annotated[
        Literal[
            "v_48_002",
            "v_48_010",
            "v_48_020",
            "v_48_030",  # vanilla models
            "ca_48_002",
            "ca_48_010",
            "ca_48_020",  # CA models
            "s_48_002",
            "s_48_010",
            "s_48_020",
            "s_48_030",  # soluble models
        ],
        typer.Option("--model", "-m", help="ProteinMPNN model to use"),
    ] = "v_48_020",
    designable_residues: Annotated[
        str,
        typer.Option(
            "--design",
            "-d",
            help="Residues to compute conditional probs for (e.g., 'A1-A68')",
        ),
    ] = "",
    unconditional: Annotated[
        bool,
        typer.Option(
            "--unconditional",
            "-u",
            help="Compute unconditional probabilities (no sequence context)",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory (defaults to PDB directory)",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
) -> None:
    """Compute amino acid conditional (or unconditional) probabilities per residue.

    Outputs both CSV and NPZ files with log probabilities for each residue position.

    Example usage:

        proteinmpnn compute-probs 6MRR.pdb --design A1-A68

        proteinmpnn compute-probs 6MRR.pdb --unconditional

    """
    # Determine output directory
    if output is None:
        output = pdb_path.parent
    output.mkdir(parents=True, exist_ok=True)

    # Create runner and compute probabilities
    logger.info("Loading model %s...", model_name)
    runner = InferenceRunner(model_name=model_name)

    mode_str = "unconditional" if unconditional else "conditional"
    logger.info("Computing %s probabilities for %s...", mode_str, pdb_path.name)

    result = runner.compute_probs(
        pdb_path=pdb_path,
        designable_res=designable_residues,
        unconditional=unconditional,
        seed=seed,
    )

    # Write outputs
    csv_path = output / f"{pdb_path.stem}_probs.csv"
    write_probs_csv(result, csv_path)
    logger.info("Wrote CSV to %s", csv_path)

    npz_path = output / f"{pdb_path.stem}_probs.npz"
    write_probs_npz(result, npz_path)
    logger.info("Wrote NPZ to %s", npz_path)

    # Print summary
    logger.info(
        "Computed %s log probabilities for %d residues",
        mode_str,
        len(result.residue_info),
    )
    logger.info("Log probability matrix shape: %s", result.log_probs.shape)

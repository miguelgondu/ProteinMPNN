"""Run single protein design CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import typer

from proteinmpnn.cli import app
from proteinmpnn.cli.output import write_af2_csv, write_fasta
from proteinmpnn.inference import InferenceRunner

if TYPE_CHECKING:
    from pathlib import Path


@app.command()
def run_single(
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
            help="Designable residues (e.g., 'A1-A68' or 'A10,A12-A15')",
        ),
    ] = "",
    symmetric_residues: Annotated[
        str,
        typer.Option(
            "--symmetric",
            "-s",
            help="Symmetric residue pairs (e.g., 'A10:B10,A11:B11')",
        ),
    ] = "",
    cluster_center: Annotated[
        str,
        typer.Option(
            "--cluster",
            "-c",
            help="Cluster center residue(s) for radius-based selection",
        ),
    ] = "",
    cluster_radius: Annotated[
        float,
        typer.Option("--radius", "-r", help="Cluster radius in Angstroms"),
    ] = 10.0,
    backbone_noise: Annotated[
        float,
        typer.Option("--noise", help="Backbone noise standard deviation"),
    ] = 0.0,
    num_seq_per_target: Annotated[
        int,
        typer.Option("-n", "--num-seqs", help="Number of sequences to generate"),
    ] = 5,
    batch_size: Annotated[
        int,
        typer.Option("--batch", "-b", help="Batch size for generation"),
    ] = 1,
    temperature: Annotated[
        str,
        typer.Option(
            "--temp",
            "-t",
            help="Sampling temperature(s), space-separated (e.g., '0.1 0.2')",
        ),
    ] = "0.1",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory (defaults to PDB directory)",
        ),
    ] = None,
    af2: Annotated[
        bool,
        typer.Option("--af2", help="Also output AlphaFold2-format CSV"),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
) -> None:
    """Design sequences for a single protein structure.

    Example usage:

        proteinmpnn run-single 6MRR.pdb --design A1-A68 -n 10

        proteinmpnn run-single 4GYT.pdb --design A7-A183,B7-B183 \\
            --symmetric A7-A183:B7-B183

    """
    # Parse temperatures
    temperatures = [float(t) for t in temperature.split()]

    # Determine output directory
    if output is None:
        output = pdb_path.parent
    output.mkdir(parents=True, exist_ok=True)

    # Create runner and generate sequences
    typer.echo(f"Loading model {model_name}...")
    runner = InferenceRunner(
        model_name=model_name,
        backbone_noise=backbone_noise,
    )

    typer.echo(f"Designing sequences for {pdb_path.name}...")
    result = runner.design_single(
        pdb_path=pdb_path,
        designable_res=designable_residues,
        symmetric_res=symmetric_residues,
        cluster_center=cluster_center,
        cluster_radius=cluster_radius,
        num_sequences=num_seq_per_target,
        batch_size=batch_size,
        temperatures=temperatures,
        seed=seed,
    )

    # Write outputs
    fasta_path = output / f"{pdb_path.stem}.fasta"
    write_fasta(result, fasta_path)
    typer.echo(f"Wrote {len(result.sequences) + 1} sequences to {fasta_path}")

    if af2:
        csv_path = output / f"{pdb_path.stem}.csv"
        write_af2_csv(result, csv_path)
        typer.echo(f"Wrote AlphaFold2 CSV to {csv_path}")

    # Print summary
    typer.echo(f"\nNative sequence score: {result.native.score:.4f}")
    best_score = min(s.score for s in result.sequences)
    typer.echo(f"Best designed sequence score: {best_score:.4f}")

"""CLI for ProteinMPNN."""

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)

# Import commands to register them
from proteinmpnn.cli import (  # noqa: F401, E402
    compute_probs,  # noqa: F401, E402
    run_single,  # noqa: F401, E402
)

if __name__ == "__main__":
    app()

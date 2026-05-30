"""CLI for ProteinMPNN."""

import logging
from typing import Annotated

import typer

from proteinmpnn.utils.logging import setup_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            help="Logging level",
            case_sensitive=False,
        ),
    ] = "INFO",
) -> None:
    """ProteinMPNN: Structure-conditioned protein sequence design."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    setup_logging(level=level)


# Import commands to register them
from proteinmpnn.cli import (  # noqa: F401, E402
    compute_probs,  # noqa: F401, E402
    run_single,  # noqa: F401, E402
)

if __name__ == "__main__":
    app()

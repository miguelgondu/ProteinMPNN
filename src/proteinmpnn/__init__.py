"""ProteinMPNN - Protein sequence design using message passing neural networks."""

import logging
from importlib.metadata import version

# Prevent "No handler found" warnings for library users who don't configure logging
logging.getLogger("proteinmpnn").addHandler(logging.NullHandler())

__version__ = version("proteinmpnn-cli")

# Re-export main classes for convenient imports
from proteinmpnn.inference import (  # noqa: E402
    DesignResult,
    InferenceRunner,
    SequenceResult,
)

__all__ = [
    "__version__",
    "InferenceRunner",
    "DesignResult",
    "SequenceResult",
]


def main() -> None:
    """Entry point for the CLI."""
    from proteinmpnn.cli import app

    app()

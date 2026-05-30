"""ProteinMPNN - Protein sequence design using message passing neural networks."""

import logging

# Prevent "No handler found" warnings for library users who don't configure logging
logging.getLogger("proteinmpnn").addHandler(logging.NullHandler())

# Re-export main classes for convenient imports
from proteinmpnn.inference import DesignResult, InferenceRunner, SequenceResult

__all__ = [
    "InferenceRunner",
    "DesignResult",
    "SequenceResult",
]


def main() -> None:
    """Entry point for the CLI."""
    from proteinmpnn.cli import app

    app()

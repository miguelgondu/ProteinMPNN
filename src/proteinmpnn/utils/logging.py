"""Logging utilities for ProteinMPNN."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure the root logger for the proteinmpnn package.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to a file for logging output.
        log_format: Optional custom format string. If None, uses a default format
            with timestamp, level, logger name, and message.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handlers: list[logging.Handler] = []

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    )
    handlers.append(console_handler)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)

    # Configure the proteinmpnn logger
    logger = logging.getLogger("proteinmpnn")
    logger.setLevel(level)
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: The name of the module. If None, returns the root proteinmpnn logger.
            If provided, creates a child logger under proteinmpnn (e.g., "model"
            becomes "proteinmpnn.model").

    Returns:
        A configured logger instance.
    """
    if name is None:
        return logging.getLogger("proteinmpnn")
    return logging.getLogger(f"proteinmpnn.{name}")

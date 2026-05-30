"""Shared residue parsing utilities."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from proteinmpnn.utils.constants import CHAIN_IDS


def parse_residue(res_item: str) -> tuple[str, int]:
    """Parse a residue string like 'A10' into (chain, index).

    Args:
        res_item: A residue identifier in the format "ChainResidueNumber"
            (e.g., "A10", "B123").

    Returns:
        A tuple of (chain_id, residue_index).

    Raises:
        ValueError: If the residue string cannot be parsed or contains
            an unknown chain ID.

    Examples:
        >>> parse_residue("A10")
        ('A', 10)
        >>> parse_residue("B123")
        ('B', 123)
    """
    split_item = [item for item in re.split(r"(\d+)", res_item) if item]

    if len(split_item) != 2:
        raise ValueError(f"Unable to parse residue: {res_item}.")
    if split_item[0] not in CHAIN_IDS:
        raise ValueError(f"Unknown chain id in residue: {res_item}")
    return (split_item[0], int(split_item[1]))


def parse_residue_range(range_item: str) -> list[tuple[str, int]]:
    """Parse a range like 'A10-A15' into list of (chain, index) tuples.

    Args:
        range_item: A residue range in the format "ChainStart-ChainEnd"
            (e.g., "A10-A15").

    Returns:
        A list of (chain_id, residue_index) tuples for all residues
        in the range (inclusive).

    Raises:
        ValueError: If the range cannot be parsed, spans multiple chains,
            or has invalid indices.

    Examples:
        >>> parse_residue_range("A10-A12")
        [('A', 10), ('A', 11), ('A', 12)]
    """
    split_range = range_item.split("-")
    if len(split_range) != 2:
        raise ValueError(f"Unable to parse residue range: {range_item}")

    start_item, finish_item = split_range[0], split_range[1]

    s_chain, s_idx = parse_residue(start_item)
    f_chain, f_idx = parse_residue(finish_item)

    if s_chain != f_chain:
        raise ValueError(f"Residue ranges cannot span multiple chains: {range_item}")
    if s_idx >= f_idx:
        raise ValueError(
            f"Residue range start must be smaller than end: {range_item}"
        )

    return [(s_chain, i) for i in range(s_idx, f_idx + 1)]


def parse_residue_list(res_str: str) -> list[tuple[str, int]]:
    """Parse comma-separated residues/ranges like 'A10,A12-A15'.

    Args:
        res_str: A comma-separated string of residue identifiers and/or ranges.

    Returns:
        A list of (chain_id, residue_index) tuples for all specified residues.

    Examples:
        >>> parse_residue_list("A10,A12-A14")
        [('A', 10), ('A', 12), ('A', 13), ('A', 14)]
        >>> parse_residue_list("B5")
        [('B', 5)]
    """
    items = [s for s in res_str.strip().split(",") if s]

    result: list[tuple[str, int]] = []
    for item in items:
        if "-" in item:
            result.extend(parse_residue_range(item))
        else:
            result.append(parse_residue(item))

    return result


def validate_symmetric_groups(
    groups: Sequence[Sequence[tuple[str, int]]],
) -> None:
    """Validate that symmetric groups have equal lengths.

    Args:
        groups: A sequence of sequences of residue tuples to validate.

    Raises:
        ValueError: If the groups have different lengths.
    """
    if not groups:
        return

    lengths = [len(g) for g in groups]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Tied residues/ranges must have same size for symmetry. "
            f"Got lengths: {lengths}"
        )

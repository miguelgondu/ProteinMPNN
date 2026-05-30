"""Tests for residue parsing utilities."""

from __future__ import annotations

import pytest

from proteinmpnn.utils.residue import (
    parse_residue,
    parse_residue_list,
    parse_residue_range,
    validate_symmetric_groups,
)


class TestParseResidue:
    """Tests for parse_residue function."""

    def test_simple_residue(self) -> None:
        assert parse_residue("A10") == ("A", 10)

    def test_three_digit_residue(self) -> None:
        assert parse_residue("B123") == ("B", 123)

    def test_single_digit_residue(self) -> None:
        assert parse_residue("Z1") == ("Z", 1)

    def test_lowercase_chain(self) -> None:
        assert parse_residue("a5") == ("a", 5)

    def test_invalid_format_raises(self) -> None:
        # "10A" splits to ["10", "A"] - "10" is not a valid chain ID
        with pytest.raises(ValueError, match="Unknown chain id"):
            parse_residue("10A")

    def test_numbers_only_raises(self) -> None:
        # "110" splits to just ["110"] - not 2 parts
        with pytest.raises(ValueError, match="Unable to parse residue"):
            parse_residue("110")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unable to parse residue"):
            parse_residue("")


class TestParseResidueRange:
    """Tests for parse_residue_range function."""

    def test_simple_range(self) -> None:
        result = parse_residue_range("A10-A12")
        assert result == [("A", 10), ("A", 11), ("A", 12)]

    def test_single_element_range(self) -> None:
        result = parse_residue_range("B5-B6")
        assert result == [("B", 5), ("B", 6)]

    def test_larger_range(self) -> None:
        result = parse_residue_range("C1-C5")
        assert result == [("C", 1), ("C", 2), ("C", 3), ("C", 4), ("C", 5)]

    def test_cross_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot span multiple chains"):
            parse_residue_range("A10-B15")

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError, match="start must be smaller than end"):
            parse_residue_range("A15-A10")

    def test_same_index_raises(self) -> None:
        with pytest.raises(ValueError, match="start must be smaller than end"):
            parse_residue_range("A10-A10")

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unable to parse residue range"):
            parse_residue_range("A10-A15-A20")


class TestParseResidueList:
    """Tests for parse_residue_list function."""

    def test_single_residue(self) -> None:
        result = parse_residue_list("B5")
        assert result == [("B", 5)]

    def test_multiple_residues(self) -> None:
        result = parse_residue_list("A1,B2,C3")
        assert result == [("A", 1), ("B", 2), ("C", 3)]

    def test_mixed_residues_and_ranges(self) -> None:
        result = parse_residue_list("A10,A12-A14")
        assert result == [("A", 10), ("A", 12), ("A", 13), ("A", 14)]

    def test_multiple_ranges(self) -> None:
        result = parse_residue_list("A1-A2,B1-B2")
        assert result == [("A", 1), ("A", 2), ("B", 1), ("B", 2)]

    def test_leading_trailing_whitespace(self) -> None:
        # Leading/trailing whitespace on the whole string is handled
        result = parse_residue_list("  A1,A2,A3  ")
        assert result == [("A", 1), ("A", 2), ("A", 3)]

    def test_empty_string(self) -> None:
        result = parse_residue_list("")
        assert result == []


class TestValidateSymmetricGroups:
    """Tests for validate_symmetric_groups function."""

    def test_valid_equal_lengths(self) -> None:
        groups = [[("A", 1), ("A", 2)], [("B", 1), ("B", 2)]]
        validate_symmetric_groups(groups)  # Should not raise

    def test_valid_single_elements(self) -> None:
        groups = [[("A", 1)], [("B", 1)], [("C", 1)]]
        validate_symmetric_groups(groups)  # Should not raise

    def test_empty_groups(self) -> None:
        validate_symmetric_groups([])  # Should not raise

    def test_unequal_lengths_raises(self) -> None:
        groups = [[("A", 1), ("A", 2)], [("B", 1)]]
        with pytest.raises(ValueError, match="same size"):
            validate_symmetric_groups(groups)

    def test_multiple_unequal_lengths_raises(self) -> None:
        groups = [[("A", 1)], [("B", 1), ("B", 2)], [("C", 1), ("C", 2), ("C", 3)]]
        with pytest.raises(ValueError, match="same size"):
            validate_symmetric_groups(groups)

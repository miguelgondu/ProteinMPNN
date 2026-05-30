"""Tests for the transform module."""

from __future__ import annotations

import pytest

from proteinmpnn.data.config import DesignableResidue, SingleStateConfig
from proteinmpnn.inference.transform import (
    make_fixed_positions_dict,
    make_omit_aa_dict,
    make_tied_positions_dict,
    parse_mutate_to,
    transform_inputs,
)


class TestParseMutateTo:
    """Tests for parse_mutate_to function."""

    def test_mutate_to_all_returns_only_x(self):
        """When MutTo is 'all', should only omit X."""
        res = {"MutTo": "all", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=False)
        assert result == "X"

    def test_mutate_to_hydphob_omits_hydrophilics(self):
        """Hydphob should omit hydrophilic amino acids."""
        res = {"MutTo": "hydphob", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=False)
        # Should omit CDEHKNPQRSTX
        for aa in "CDEHKNPQRST":
            assert aa in result

    def test_mutate_to_hydphil_omits_hydrophobics(self):
        """Hydphil should omit hydrophobic amino acids."""
        res = {"MutTo": "hydphil", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=False)
        # Should omit ACFGILMPVWYX
        for aa in "ACFGILMPVWY":
            assert aa in result

    def test_mutate_to_specific_aas(self):
        """Specific AAs should omit everything else."""
        res = {"MutTo": "AV", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=False)
        # Should NOT have A or V in omit list
        assert "A" not in result
        assert "V" not in result

    def test_mutate_to_all_minus_c(self):
        """all-C should omit only C and X."""
        res = {"MutTo": "all-C", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=False)
        assert "C" in result
        assert "X" in result

    def test_experimental_hydphob_plus_c(self):
        """Experimental: hydphob+C should include C in allowed."""
        res = {"MutTo": "hydphob+C", "WTAA": "A"}
        result = parse_mutate_to(res, experimental=True)
        # C should NOT be in omit list when explicitly added
        assert "C" not in result


class TestMakeFixedPositionsDict:
    """Tests for make_fixed_positions_dict function."""

    @pytest.fixture
    def mock_protein(self) -> dict:
        """Create a mock protein dictionary."""
        return {
            "name": "test_protein",
            "seq_chain_A": "ACDEFGHIKLMNPQRSTVWY",  # 20 residues
            "seq_chain_B": "ACDEFGHIKL",  # 10 residues
            "num_of_chains": 2,
        }

    def test_all_designable_returns_none(self, mock_protein):
        """When all positions are designable, should return None."""
        config = SingleStateConfig(
            sequence={"A": mock_protein["seq_chain_A"]},
            designable=[
                DesignableResidue(chain="A", resid=i, WTAA="A", MutTo="all")
                for i in range(1, 21)
            ],
            symmetric=[],
        )
        result = make_fixed_positions_dict(config, mock_protein)
        # Chain B should still be fixed
        assert result is not None
        assert "B" in result["test_protein"]

    def test_partial_design_has_fixed_positions(self, mock_protein):
        """Partially designable should have fixed positions."""
        config = SingleStateConfig(
            sequence={"A": mock_protein["seq_chain_A"]},
            designable=[
                DesignableResidue(chain="A", resid=1, WTAA="A", MutTo="all"),
                DesignableResidue(chain="A", resid=5, WTAA="F", MutTo="all"),
            ],
            symmetric=[],
        )
        result = make_fixed_positions_dict(config, mock_protein)
        assert result is not None
        fixed_A = result["test_protein"]["A"]
        # Positions 1 and 5 should NOT be in fixed list
        assert 1 not in fixed_A
        assert 5 not in fixed_A
        # But positions 2, 3, 4 should be fixed
        assert 2 in fixed_A
        assert 3 in fixed_A
        assert 4 in fixed_A


class TestMakeTiedPositionsDict:
    """Tests for make_tied_positions_dict function."""

    @pytest.fixture
    def mock_protein(self) -> dict:
        """Create a mock protein dictionary."""
        return {
            "name": "test_protein",
            "seq_chain_A": "ACDEFGHIKLMNPQRSTVWY",
            "seq_chain_B": "ACDEFGHIKLMNPQRSTVWY",
            "num_of_chains": 2,
        }

    def test_no_symmetric_returns_none(self, mock_protein):
        """When no symmetric constraints, should return None."""
        config = SingleStateConfig(
            sequence={"A": "ACDEF", "B": "ACDEF"},
            designable=[],
            symmetric=[],
        )
        result = make_tied_positions_dict(config, mock_protein)
        assert result is None

    def test_symmetric_creates_tied_dict(self, mock_protein):
        """Symmetric constraints should create tied positions."""
        config = SingleStateConfig(
            sequence={"A": "ACDEF", "B": "ACDEF"},
            designable=[],
            symmetric=[["A1", "B1"], ["A2", "B2"]],
        )
        result = make_tied_positions_dict(config, mock_protein)
        assert result is not None
        tied_list = result["test_protein"]
        assert len(tied_list) == 2
        # First tie should have A:1 and B:1
        assert {"A": [1], "B": [1]} in tied_list


class TestMakeOmitAADict:
    """Tests for make_omit_aa_dict function."""

    @pytest.fixture
    def mock_protein(self) -> dict:
        """Create a mock protein dictionary."""
        return {
            "name": "test_protein",
            "seq_chain_A": "ACDEFGHIKL",
            "num_of_chains": 1,
        }

    def test_all_mutatable_has_empty_omit(self, mock_protein):
        """When MutTo is 'all', should have empty omit for that position."""
        config = SingleStateConfig(
            sequence={"A": "ACDEFGHIKL"},
            designable=[
                DesignableResidue(chain="A", resid=1, WTAA="A", MutTo="all"),
            ],
            symmetric=[],
        )
        result = make_omit_aa_dict(config, mock_protein)
        # Position 1 with MutTo='all' should not be in the omit list
        assert result["test_protein"]["A"] == []

    def test_restricted_mutation_has_omit(self, mock_protein):
        """When MutTo is restricted, should have omit AAs."""
        config = SingleStateConfig(
            sequence={"A": "ACDEFGHIKL"},
            designable=[
                DesignableResidue(chain="A", resid=1, WTAA="A", MutTo="hydphob"),
            ],
            symmetric=[],
        )
        result = make_omit_aa_dict(config, mock_protein)
        assert len(result["test_protein"]["A"]) == 1
        resid, omit_aas = result["test_protein"]["A"][0]
        assert resid == 1
        assert len(omit_aas) > 0


class TestTransformInputs:
    """Tests for the main transform_inputs function."""

    @pytest.fixture
    def mock_protein(self) -> dict:
        """Create a mock protein dictionary."""
        return {
            "name": "test_protein",
            "seq_chain_A": "ACDEFGHIKL",
            "num_of_chains": 1,
        }

    def test_returns_7_tuple(self, mock_protein):
        """transform_inputs should return a 7-tuple."""
        config = SingleStateConfig(
            sequence={"A": "ACDEFGHIKL"},
            designable=[],
            symmetric=[],
        )
        result = transform_inputs(config, mock_protein)
        assert len(result) == 7

    def test_chain_id_dict_is_none(self, mock_protein):
        """chain_id_dict should always be None."""
        config = SingleStateConfig(
            sequence={"A": "ACDEFGHIKL"},
            designable=[],
            symmetric=[],
        )
        result = transform_inputs(config, mock_protein)
        assert result[0] is None

    def test_pssm_dict_is_none(self, mock_protein):
        """pssm_dict should always be None."""
        config = SingleStateConfig(
            sequence={"A": "ACDEFGHIKL"},
            designable=[],
            symmetric=[],
        )
        result = transform_inputs(config, mock_protein)
        assert result[2] is None

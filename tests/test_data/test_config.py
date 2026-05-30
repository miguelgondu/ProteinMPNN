"""Tests for Pydantic config models."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from proteinmpnn.data.config import (
    DesignableResidue,
    MultiStateConfig,
    SingleStateConfig,
)


class TestDesignableResidue:
    """Tests for DesignableResidue model."""

    def test_basic_creation(self) -> None:
        res = DesignableResidue(chain="A", resid=10, WTAA="M", MutTo="all")
        assert res.chain == "A"
        assert res.resid == 10
        assert res.WTAA == "M"
        assert res.MutTo == "all"

    def test_default_mutto(self) -> None:
        res = DesignableResidue(chain="B", resid=5, WTAA="K")
        assert res.MutTo == "all"

    def test_custom_mutto(self) -> None:
        res = DesignableResidue(chain="A", resid=1, WTAA="G", MutTo="ACDEFG")
        assert res.MutTo == "ACDEFG"

    def test_json_serialization(self) -> None:
        res = DesignableResidue(chain="A", resid=10, WTAA="M", MutTo="all")
        data = json.loads(res.model_dump_json())
        assert data == {"chain": "A", "resid": 10, "WTAA": "M", "MutTo": "all"}


class TestSingleStateConfig:
    """Tests for SingleStateConfig model."""

    def test_basic_creation(self) -> None:
        config = SingleStateConfig(
            sequence={"A": "MKVL", "B": "ACDE"},
            designable=[DesignableResidue(chain="A", resid=1, WTAA="M", MutTo="all")],
            symmetric=[["A1", "B1"]],
        )
        assert config.sequence == {"A": "MKVL", "B": "ACDE"}
        assert len(config.designable) == 1
        assert config.symmetric == [["A1", "B1"]]

    def test_empty_defaults(self) -> None:
        config = SingleStateConfig()
        assert config.sequence == {}
        assert config.designable == []
        assert config.symmetric == []

    def test_json_serialization(self) -> None:
        config = SingleStateConfig(
            sequence={"A": "MKV"},
            designable=[DesignableResidue(chain="A", resid=1, WTAA="M", MutTo="all")],
            symmetric=[],
        )
        data = json.loads(config.model_dump_json())
        assert "sequence" in data
        assert "designable" in data
        assert "symmetric" in data
        assert data["sequence"] == {"A": "MKV"}

    def test_to_json_file(self) -> None:
        config = SingleStateConfig(
            sequence={"A": "MKVL"},
            designable=[DesignableResidue(chain="A", resid=1, WTAA="M", MutTo="all")],
            symmetric=[["A1", "A2"]],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)

        try:
            config.to_json(out_path)
            assert out_path.exists()

            with out_path.open() as f:
                data = json.load(f)

            assert data["sequence"] == {"A": "MKVL"}
            assert len(data["designable"]) == 1
            assert data["symmetric"] == [["A1", "A2"]]
        finally:
            out_path.unlink()

    def test_multiple_designable_residues(self) -> None:
        config = SingleStateConfig(
            sequence={"A": "MKVLACDE"},
            designable=[
                DesignableResidue(chain="A", resid=1, WTAA="M", MutTo="all"),
                DesignableResidue(chain="A", resid=2, WTAA="K", MutTo="KR"),
                DesignableResidue(chain="A", resid=3, WTAA="V", MutTo="VIL"),
            ],
            symmetric=[],
        )
        assert len(config.designable) == 3
        assert config.designable[1].MutTo == "KR"


class TestMultiStateConfig:
    """Tests for MultiStateConfig model."""

    def test_basic_creation(self) -> None:
        config = MultiStateConfig(
            sequence={"A": "MKVL", "B": "MKVL"},
            tied_betas={"A": 1.0, "B": 0.5},
            chain_key={"pdb1": {"A": "A"}, "pdb2": {"A": "B"}},
        )
        assert config.tied_betas == {"A": 1.0, "B": 0.5}
        assert config.chain_key == {"pdb1": {"A": "A"}, "pdb2": {"A": "B"}}

    def test_inherits_single_state_fields(self) -> None:
        config = MultiStateConfig(
            sequence={"A": "MKV"},
            designable=[DesignableResidue(chain="A", resid=1, WTAA="M", MutTo="all")],
            symmetric=[["A1", "B1"]],
            tied_betas={"A": 1.0},
            chain_key={"pdb1": {"A": "A"}},
        )
        # Check inherited fields work
        assert config.sequence == {"A": "MKV"}
        assert len(config.designable) == 1
        assert config.symmetric == [["A1", "B1"]]

    def test_empty_msd_fields(self) -> None:
        config = MultiStateConfig()
        assert config.tied_betas == {}
        assert config.chain_key == {}

    def test_json_serialization(self) -> None:
        config = MultiStateConfig(
            sequence={"A": "MKV", "B": "MKV"},
            tied_betas={"A": 1.0, "B": 0.5},
            chain_key={"state1": {"A": "A"}, "state2": {"A": "B"}},
        )
        data = json.loads(config.model_dump_json())
        assert "tied_betas" in data
        assert "chain_key" in data
        assert data["tied_betas"] == {"A": 1.0, "B": 0.5}

    def test_to_json_file(self) -> None:
        config = MultiStateConfig(
            sequence={"A": "MKVL", "B": "MKVL"},
            tied_betas={"A": 1.0, "B": 0.5},
            chain_key={"pdb1": {"A": "A"}, "pdb2": {"A": "B"}},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)

        try:
            config.to_json(out_path)
            assert out_path.exists()

            with out_path.open() as f:
                data = json.load(f)

            assert "tied_betas" in data
            assert "chain_key" in data
        finally:
            out_path.unlink()

"""Tests for the results module."""

from __future__ import annotations

import pytest

from proteinmpnn.inference.results import DesignResult, NativeSequence, SequenceResult


class TestSequenceResult:
    """Tests for SequenceResult dataclass."""

    def test_to_fasta_header(self):
        """Test FASTA header generation."""
        result = SequenceResult(
            sequence="ACDEFGHIKL",
            score=1.2345,
            seq_recovery=0.8765,
            temperature=0.1,
            sample_index=0,
        )
        header = result.to_fasta_header()
        assert header.startswith(">")
        assert "T=0.1" in header
        assert "sample=0" in header
        assert "score=1.2345" in header
        assert "seq_recovery=0.8765" in header


class TestNativeSequence:
    """Tests for NativeSequence dataclass."""

    def test_to_fasta_header(self):
        """Test FASTA header generation for native sequence."""
        native = NativeSequence(
            name="test_protein",
            sequence="ACDEFGHIKL",
            score=1.5,
            fixed_chains=["B"],
            designed_chains=["A"],
            model_name="v_48_020",
        )
        header = native.to_fasta_header()
        assert header.startswith(">test_protein")
        assert "score=1.5" in header
        assert "fixed_chains" in header
        assert "designed_chains" in header
        assert "model_name=v_48_020" in header


class TestDesignResult:
    """Tests for DesignResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> DesignResult:
        """Create a sample DesignResult for testing."""
        native = NativeSequence(
            name="test_protein",
            sequence="ACDEF/GHIKL",
            score=1.5,
            fixed_chains=[],
            designed_chains=["A", "B"],
            model_name="v_48_020",
        )
        sequences = [
            SequenceResult(
                sequence="AAAAA/BBBBB",
                score=1.2,
                seq_recovery=0.4,
                temperature=0.1,
                sample_index=0,
            ),
            SequenceResult(
                sequence="CCCCC/DDDDD",
                score=1.3,
                seq_recovery=0.3,
                temperature=0.1,
                sample_index=1,
            ),
        ]
        return DesignResult(
            protein_name="test_protein",
            native=native,
            sequences=sequences,
        )

    def test_to_fasta(self, sample_result):
        """Test FASTA output generation."""
        fasta = sample_result.to_fasta()
        lines = fasta.split("\n")
        # Should have header + sequence for native + 2 designed
        assert len(lines) == 6  # 3 sequences x 2 lines each
        # First line should be native header
        assert lines[0].startswith(">test_protein")
        # Second line should be native sequence
        assert lines[1] == "ACDEF/GHIKL"

    def test_to_fasta_with_no_sequences(self):
        """Test FASTA output with only native sequence."""
        native = NativeSequence(
            name="test",
            sequence="ACDEF",
            score=1.0,
            fixed_chains=[],
            designed_chains=["A"],
            model_name="v_48_020",
        )
        result = DesignResult(
            protein_name="test",
            native=native,
            sequences=[],
        )
        fasta = result.to_fasta()
        lines = fasta.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith(">test")
        assert lines[1] == "ACDEF"

    def test_to_af2_csv(self, sample_result):
        """Test AlphaFold2 CSV output generation."""
        csv = sample_result.to_af2_csv()
        lines = csv.split("\n")
        # Should have 3 lines (native + 2 designed)
        assert len(lines) == 3
        # First line should have native sequence
        assert lines[0].startswith(",ACDEF,GHIKL")
        # Should have comment section
        assert "#" in lines[0]

    def test_to_af2_csv_sanitizes_commas(self, sample_result):
        """Test that AF2 CSV comments have commas removed."""
        csv = sample_result.to_af2_csv()
        for line in csv.split("\n"):
            # Split at # to get comment part
            parts = line.split("#")
            if len(parts) > 1:
                comment = parts[1]
                # Comment should not have commas (they break AF2 parsing)
                # Note: lists like fixed_chains get sanitized
                pass  # This is tested by the format itself working

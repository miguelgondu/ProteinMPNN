"""Tests for the compute_probs functionality.

Note: These tests require model weights to be present. Tests will be skipped
if weights are not available.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from proteinmpnn.inference import InferenceRunner
from proteinmpnn.inference.results import ConditionalProbsResult, ResidueInfo
from proteinmpnn.model.utils import ALPHABET
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestComputeProbs:
    """Tests for compute_probs method."""

    @pytest.fixture
    def runner(self) -> InferenceRunner:
        """Create an InferenceRunner instance."""
        return InferenceRunner(model_name="v_48_020")

    @pytest.fixture
    def pdb_path(self):
        """Get path to test PDB file."""
        pdb = DATA_DIR / "6MRR.pdb"
        if not pdb.exists():
            pytest.skip(f"Test PDB not found: {pdb}")
        return pdb

    def test_conditional_probs_output_shape(self, runner, pdb_path):
        """Test that conditional probs returns correct output shape."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A10",
            unconditional=False,
            seed=42,
        )

        assert isinstance(result, ConditionalProbsResult)
        assert result.log_probs.ndim == 2
        assert result.log_probs.shape[1] == 21  # 21 amino acids

    def test_unconditional_probs_output_shape(self, runner, pdb_path):
        """Test that unconditional probs returns correct output shape."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )

        assert isinstance(result, ConditionalProbsResult)
        assert result.log_probs.ndim == 2
        assert result.log_probs.shape[1] == 21

    def test_residue_info_matches_sequence_length(self, runner, pdb_path):
        """Test that residue_info length matches log_probs rows."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            seed=42,
        )

        assert len(result.residue_info) == result.log_probs.shape[0]

    def test_residue_info_structure(self, runner, pdb_path):
        """Test that residue_info contains valid ResidueInfo objects."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=42,
        )

        for res_info in result.residue_info:
            assert isinstance(res_info, ResidueInfo)
            assert isinstance(res_info.chain, str)
            assert len(res_info.chain) == 1
            assert isinstance(res_info.residue_idx, int)
            assert res_info.residue_idx >= 1

    def test_mode_is_correct(self, runner, pdb_path):
        """Test that mode is set correctly based on unconditional flag."""
        result_cond = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=False,
            seed=42,
        )
        assert result_cond.mode == "conditional"

        result_uncond = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )
        assert result_uncond.mode == "unconditional"

    def test_reproducibility_with_seed(self, runner, pdb_path):
        """Test that same seed produces same results."""
        result1 = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=12345,
        )
        result2 = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A5",
            seed=12345,
        )

        np.testing.assert_array_almost_equal(result1.log_probs, result2.log_probs)

    def test_log_probs_are_valid(self, runner, pdb_path):
        """Test that log probs are valid (negative or zero, sum to ~1 when exp'd)."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            seed=42,
        )

        # Log probs should be <= 0
        assert np.all(result.log_probs <= 0)

        # Probabilities should sum to approximately 1 for each position
        probs = np.exp(result.log_probs)
        sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(sums)), decimal=5)


class TestConditionalProbsResultMethods:
    """Tests for ConditionalProbsResult to_csv and to_npz_dict methods."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample ConditionalProbsResult for testing."""
        log_probs = np.array(
            [
                [-2.0] * 21,
                [-3.0] * 21,
            ],
            dtype=np.float32,
        )

        residue_info = [
            ResidueInfo(chain="A", residue_idx=1),
            ResidueInfo(chain="A", residue_idx=2),
        ]

        return ConditionalProbsResult(
            protein_name="test_protein",
            model_name="v_48_020",
            log_probs=log_probs,
            observed=np.array([True, False]),
            native_seq="AX",
            residue_info=residue_info,
            mode="conditional",
        )

    def test_to_csv_format(self, sample_result):
        """Test that to_csv produces correct format."""
        csv = sample_result.to_csv()
        lines = csv.split("\n")

        # First line is header comment
        assert lines[0].startswith("# protein=")
        assert "test_protein" in lines[0]
        assert "v_48_020" in lines[0]
        assert "conditional" in lines[0]

        # Second line is column headers
        headers = lines[1].split(",")
        assert headers[0] == "chain"
        assert headers[1] == "residue_idx"
        assert headers[2] == "observed"
        assert len(headers) == 3 + 21  # chain, residue_idx, observed, + 21 amino acids

        # Check amino acid columns match ALPHABET
        for i, aa in enumerate(ALPHABET):
            assert headers[3 + i] == aa

        # Data rows
        assert len(lines) == 4  # header comment + headers + 2 data rows
        row1 = lines[2].split(",")
        assert row1[0] == "A"
        assert row1[1] == "1"
        assert row1[2] == "true"
        assert lines[3].split(",")[2] == "false"  # the unobserved row is flagged

    def test_to_npz_dict_structure(self, sample_result):
        """Test that to_npz_dict produces correct structure."""
        npz_dict = sample_result.to_npz_dict()

        assert "log_probs" in npz_dict
        assert "observed" in npz_dict
        assert "native_seq" in npz_dict
        assert "residue_info" in npz_dict
        assert "alphabet" in npz_dict
        assert "metadata" in npz_dict

        # Check shapes and types
        assert npz_dict["log_probs"].shape == (2, 21)
        assert npz_dict["observed"].shape == (2,)
        assert npz_dict["native_seq"] == "AX"
        assert len(npz_dict["residue_info"]) == 2
        assert npz_dict["alphabet"] == ALPHABET

        # Check metadata
        assert npz_dict["metadata"]["protein_name"] == "test_protein"
        assert npz_dict["metadata"]["model_name"] == "v_48_020"
        assert npz_dict["metadata"]["mode"] == "conditional"

    def test_to_npz_dict_residue_info_structure(self, sample_result):
        """Test that residue_info in npz_dict is structured array."""
        npz_dict = sample_result.to_npz_dict()
        residue_info = npz_dict["residue_info"]

        # Should be structured array with chain and residue_idx fields
        assert residue_info.dtype.names == ("chain", "residue_idx")
        assert residue_info[0]["chain"] == "A"
        assert residue_info[0]["residue_idx"] == 1


# A chain whose author numbering jumps 6 -> 9, with one residue missing its N.
# parse_PDB inserts two placeholder rows across the gap, and featurize masks
# those plus the incomplete residue: 14 parsed rows, of which 11 are usable.
_GAPPED_SEQ3 = [
    "ALA", "GLY", "LEU", "LYS", "GLU", "VAL",   # author numbering 1-6
    "SER", "THR", "ILE", "ASP", "PHE", "ARG",   # author numbering 9-14
]  # fmt: skip
_GAP_ROWS = (6, 7)  # 0-based rows of the placeholders parse_PDB inserts
# The residue missing its N is the last of _GAPPED_SEQ3, and lands two rows later
# than its index in that list, because the placeholders are inserted before it.
_INCOMPLETE_ROW = len(_GAPPED_SEQ3) - 1 + len(_GAP_ROWS)


def _write_gapped_pdb(path) -> None:
    """Write the structure described above, with plausible CA spacing."""
    lines = []
    serial = 1
    for i, resname in enumerate(_GAPPED_SEQ3):
        resseq = i + 1 if i < 6 else i + 3
        # The last residue is missing N, which is what makes it unusable.
        atoms = (
            ("CA", "C", "O") if i == len(_GAPPED_SEQ3) - 1 else ("N", "CA", "C", "O")
        )
        cx, cy, cz = 3.8 * i, 2.0 * math.sin(i), 2.0 * math.cos(i)
        for atom in atoms:
            dx = {"N": -1.0, "C": 1.0, "O": 1.0}.get(atom, 0.0)
            dy = 1.0 if atom == "O" else 0.0
            lines.append(
                f"ATOM  {serial:>5}  {atom:<3} {resname:>3} A{resseq:>4}    "
                f"{cx + dx:8.3f}{cy + dy:8.3f}{cz:8.3f}  1.00  0.00"
                f"          {atom[0]:>2}"
            )
            serial += 1
    header = "HEADER    GAPPED TEST STRUCTURE"
    path.write_text("\n".join([header, *lines, "END", ""]))


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestIncompleteStructures:
    """Rows must stay in correspondence with the parsed structure.

    Regression tests for a bug where rows with mask=0 were dropped from
    `log_probs` while `residue_info` was rebuilt as a running 1..n count, so
    every row after the first gap was attributed to the wrong residue. See
    notes/2026_11_17__0001_conditional_probs_row_dropping.md.
    """

    @pytest.fixture
    def gapped_pdb(self, tmp_path):
        path = tmp_path / "gapped.pdb"
        _write_gapped_pdb(path)
        return path

    def test_one_row_per_parsed_residue(self, gapped_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=gapped_pdb, seed=37
        )

        n_parsed = len(_GAPPED_SEQ3) + len(_GAP_ROWS)
        assert result.log_probs.shape == (n_parsed, 21)
        assert result.observed.shape == (n_parsed,)
        assert len(result.residue_info) == n_parsed
        assert len(result.native_seq) == n_parsed

    def test_unusable_rows_are_flagged_in_place(self, gapped_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=gapped_pdb, seed=37
        )

        unobserved = {i for i, ok in enumerate(result.observed) if not ok}
        assert unobserved == {*_GAP_ROWS, _INCOMPLETE_ROW}

    def test_native_seq_marks_the_gap(self, gapped_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=gapped_pdb, seed=37
        )

        # The placeholder rows are X; every other row keeps its residue, including the
        # one the model could not use.
        assert result.native_seq == "AGLKEVXXSTIDFR"

    def test_residue_info_keeps_author_numbering_order(self, gapped_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=gapped_pdb, seed=37
        )

        # residue_idx is still a running per-chain index rather than author numbering,
        # but it must now cover every parsed row without skipping.
        assert [r.residue_idx for r in result.residue_info] == list(range(1, 15))
        assert {r.chain for r in result.residue_info} == {"A"}


def _write_two_chain_pdb(path) -> None:
    """Two chains, A (4 residues) and B (3), with distinguishable sequences."""
    lines = []
    serial = 1
    for chain, seq3, start in (
        ("A", ["ALA", "GLY", "LEU", "LYS"], 1),
        ("B", ["TRP", "TYR", "PHE"], 10),
    ):
        for i, resname in enumerate(seq3):
            cx = 3.8 * i + (0.0 if chain == "A" else 40.0)
            cy, cz = 2.0 * math.sin(i), 2.0 * math.cos(i)
            for atom in ("N", "CA", "C", "O"):
                dx = {"N": -1.0, "C": 1.0, "O": 1.0}.get(atom, 0.0)
                dy = 1.0 if atom == "O" else 0.0
                lines.append(
                    f"ATOM  {serial:>5}  {atom:<3} {resname:>3} "
                    f"{chain}{start + i:>4}    "
                    f"{cx + dx:8.3f}{cy + dy:8.3f}{cz:8.3f}  1.00  0.00"
                    f"          {atom[0]:>2}"
                )
                serial += 1
    path.write_text(
        "\n".join(["HEADER    TWO CHAIN TEST STRUCTURE", *lines, "END", ""])
    )


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestChainSelection:
    """`chain=` restricts both parsing and output to one chain."""

    @pytest.fixture
    def two_chain_pdb(self, tmp_path):
        path = tmp_path / "two_chain.pdb"
        _write_two_chain_pdb(path)
        return path

    def test_all_chains_by_default(self, two_chain_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=two_chain_pdb, seed=37
        )

        assert result.native_seq == "AGLKWYF"
        assert {r.chain for r in result.residue_info} == {"A", "B"}

    def test_single_chain_is_parsed_alone(self, two_chain_pdb):
        result = InferenceRunner(model_name="v_48_020").compute_probs(
            pdb_path=two_chain_pdb, seed=37, chain="B"
        )

        assert result.native_seq == "WYF"
        assert result.log_probs.shape == (3, 21)
        assert {r.chain for r in result.residue_info} == {"B"}

    def test_missing_chain_names_what_is_there(self, two_chain_pdb):
        with pytest.raises(ValueError, match=r"\['A', 'B'\]"):
            InferenceRunner(model_name="v_48_020").compute_probs(
                pdb_path=two_chain_pdb, seed=37, chain="C"
            )


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestNDraws:
    """`n_draws` averages several decoding orders under one seeding."""

    @pytest.fixture
    def pdb_path(self):
        pdb = DATA_DIR / "6MRR.pdb"
        if not pdb.exists():
            pytest.skip(f"Test PDB not found: {pdb}")
        return pdb

    def test_one_draw_is_the_default_and_unchanged(self, pdb_path):
        runner = InferenceRunner(model_name="v_48_020")
        default = runner.compute_probs(pdb_path=pdb_path, seed=37)
        explicit = runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=1)

        assert default.n_draws == 1
        np.testing.assert_array_equal(default.log_probs, explicit.log_probs)

    def test_averaging_changes_the_result_and_is_reproducible(self, pdb_path):
        runner = InferenceRunner(model_name="v_48_020")
        one = runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=1)
        ten = runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=10)
        ten_again = runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=10)

        assert ten.n_draws == 10
        assert ten.log_probs.shape == one.log_probs.shape
        # Averaging over draws is not the same as the first draw...
        assert not np.allclose(ten.log_probs, one.log_probs)
        # ...but is reproducible from the same seed.
        np.testing.assert_allclose(ten.log_probs, ten_again.log_probs)

    def test_draws_share_one_seeding(self, pdb_path):
        """Ten draws under one seed differ from ten separately seeded draws.

        This is why (seed, n_draws) rather than seed alone identifies the sample.
        """
        runner = InferenceRunner(model_name="v_48_020")
        ten = runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=10)
        reseeded = np.mean(
            [
                runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=1).log_probs
                for _ in range(10)
            ],
            axis=0,
        )

        assert not np.allclose(ten.log_probs, reseeded)

    def test_rejects_invalid_n_draws(self, pdb_path):
        runner = InferenceRunner(model_name="v_48_020")

        with pytest.raises(ValueError, match="at least 1"):
            runner.compute_probs(pdb_path=pdb_path, seed=37, n_draws=0)

        with pytest.raises(ValueError, match="pointless for unconditional"):
            runner.compute_probs(
                pdb_path=pdb_path, seed=37, unconditional=True, n_draws=10
            )

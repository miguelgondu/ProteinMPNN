"""Tests comparing compute_probs against original protein_mpnn_run.py implementation.

This test ensures that the new compute_probs implementation produces identical
results to the original run/protein_mpnn/protein_mpnn_run.py script.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from proteinmpnn.data.single_state import SingleStateDesignInput
from proteinmpnn.data.structure.dataset import StructureDatasetPDB
from proteinmpnn.inference import InferenceRunner
from proteinmpnn.inference.transform import transform_inputs
from proteinmpnn.model.featurize import tied_featurize
from proteinmpnn.utils.constants import ROOT_DIR, WEIGHTS_PATH

if TYPE_CHECKING:
    from pathlib import Path

# Test data paths
DATA_DIR = ROOT_DIR / "data" / "pdbs"


def weights_available() -> bool:
    """Check if model weights are available."""
    return (WEIGHTS_PATH / "v_48_020.pt").exists()


def run_original_conditional_probs(
    runner: InferenceRunner,
    pdb_path: Path,
    chain_id_dict: dict | None = None,
    fixed_positions_dict: dict | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run original conditional_probs logic from protein_mpnn_run.py.

    This mirrors lines 227-237 of run/protein_mpnn/protein_mpnn_run.py.

    Returns:
        Tuple of (log_probs, mask, design_mask) as the original would save.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Load protein (matching original logic)
    pdb_dataset = StructureDatasetPDB.from_pdb_dir(pdb_path.parent)
    protein = None
    for p in pdb_dataset:
        if p["name"] == pdb_path.stem:
            protein = p
            break

    if protein is None:
        raise ValueError(f"Could not find protein {pdb_path.stem}")

    # Create batch (original does BATCH_COPIES clones)
    batch_clones = [copy.deepcopy(protein)]

    # Featurize (matching original tied_featurize call at line 206)
    (
        X,
        S,
        mask,
        _,
        chain_M,
        chain_encoding_all,
        _,
        _,
        _,
        _,
        chain_M_pos,
        _,
        residue_idx,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = tied_featurize(
        batch_clones,
        runner.device,
        chain_id_dict,
        fixed_positions_dict,
    )

    # Run conditional_probs (matching original lines 232-234)
    with torch.no_grad():
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_conditional_probs = runner.model.conditional_probs(
            X,
            S,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn_1,
            False,  # conditional_probs_only_backbone=False
        )

    # Extract outputs (matching original lines 235-237)
    log_p = log_conditional_probs.cpu().numpy()  # [B, L, 21]
    mask_out = mask[0].cpu().numpy()
    design_mask_out = (chain_M * chain_M_pos * mask)[0].cpu().numpy()

    return log_p[0], mask_out, design_mask_out


def run_original_unconditional_probs(
    runner: InferenceRunner,
    pdb_path: Path,
    chain_id_dict: dict | None = None,
    fixed_positions_dict: dict | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run original unconditional_probs logic from protein_mpnn_run.py.

    This mirrors lines 238-247 of run/protein_mpnn/protein_mpnn_run.py.

    Returns:
        Tuple of (log_probs, mask, design_mask) as the original would save.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Load protein
    pdb_dataset = StructureDatasetPDB.from_pdb_dir(pdb_path.parent)
    protein = None
    for p in pdb_dataset:
        if p["name"] == pdb_path.stem:
            protein = p
            break

    if protein is None:
        raise ValueError(f"Could not find protein {pdb_path.stem}")

    batch_clones = [copy.deepcopy(protein)]

    # Featurize
    (
        X,
        S,
        mask,
        _,
        chain_M,
        chain_encoding_all,
        _,
        _,
        _,
        _,
        chain_M_pos,
        _,
        residue_idx,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = tied_featurize(
        batch_clones,
        runner.device,
        chain_id_dict,
        fixed_positions_dict,
    )

    # Run unconditional_probs (matching original lines 243-244)
    with torch.no_grad():
        log_unconditional_probs = runner.model.unconditional_probs(
            X,
            mask,
            residue_idx,
            chain_encoding_all,
        )

    # Extract outputs (matching original lines 245-247)
    log_p = log_unconditional_probs.cpu().numpy()  # [B, L, 21]
    mask_out = mask[0].cpu().numpy()
    design_mask_out = (chain_M * chain_M_pos * mask)[0].cpu().numpy()

    return log_p[0], mask_out, design_mask_out


@pytest.mark.skipif(not weights_available(), reason="Model weights not available")
class TestAgainstOriginalImplementation:
    """Tests comparing compute_probs to original protein_mpnn_run.py logic."""

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

    def test_unconditional_probs_match_original(self, runner, pdb_path):
        """Test that unconditional probs match original implementation."""
        seed = 42

        # Run original logic
        orig_log_p, orig_mask, orig_design_mask = run_original_unconditional_probs(
            runner, pdb_path, seed=seed
        )

        # Run new implementation
        result = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=seed,
        )

        # The new implementation returns only valid (unpadded) positions
        # We need to extract the same positions from the original output
        valid_indices = orig_mask > 0
        orig_log_p_valid = orig_log_p[valid_indices]

        # Compare
        np.testing.assert_array_almost_equal(
            result.log_probs,
            orig_log_p_valid,
            decimal=5,
            err_msg="Unconditional probs don't match original implementation",
        )

    def test_conditional_probs_match_original_all_designable(self, runner, pdb_path):
        """Test conditional probs match original when all residues are designable."""
        seed = 42

        # Run original logic with no fixed positions (all designable)
        # chain_id_dict=None means all chains are "masked" (designable)
        orig_log_p, orig_mask, orig_design_mask = run_original_conditional_probs(
            runner,
            pdb_path,
            chain_id_dict=None,
            fixed_positions_dict=None,
            seed=seed,
        )

        # Run new implementation (no designable_res means all residues)
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="",  # All residues
            unconditional=False,
            seed=seed,
        )

        # Extract valid positions from original
        valid_indices = orig_mask > 0
        orig_log_p_valid = orig_log_p[valid_indices]

        # Compare
        np.testing.assert_array_almost_equal(
            result.log_probs,
            orig_log_p_valid,
            decimal=5,
            err_msg="Conditional probs (all designable) don't match original",
        )

    def test_conditional_probs_match_original_with_fixed_positions(
        self, runner, pdb_path
    ):
        """Test conditional probs match original with fixed positions."""
        seed = 42

        # Set up design input to only design A1-A30
        design_input = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res="A1-A30",
            default_design_setting="all",
        )
        config = design_input.to_config()

        # Load protein for transform_inputs
        pdb_dataset = StructureDatasetPDB.from_pdb_dir(pdb_path.parent)
        protein = None
        for p in pdb_dataset:
            if p["name"] == pdb_path.stem:
                protein = p
                break

        # Get fixed_positions_dict from our transform
        (
            chain_id_dict,
            fixed_positions_dict,
            _,
            _,
            _,
            _,
            _,
        ) = transform_inputs(config, protein)

        # Run original with these fixed positions
        orig_log_p, orig_mask, orig_design_mask = run_original_conditional_probs(
            runner,
            pdb_path,
            chain_id_dict=chain_id_dict,
            fixed_positions_dict=fixed_positions_dict,
            seed=seed,
        )

        # Run new implementation
        result = runner.compute_probs(
            pdb_path=pdb_path,
            designable_res="A1-A30",
            unconditional=False,
            seed=seed,
        )

        # Extract valid positions from original
        valid_indices = orig_mask > 0
        orig_log_p_valid = orig_log_p[valid_indices]

        # Compare
        np.testing.assert_array_almost_equal(
            result.log_probs,
            orig_log_p_valid,
            decimal=5,
            err_msg="Conditional probs (A1-A30 designable) don't match original",
        )

    def test_residue_count_matches(self, runner, pdb_path):
        """Test that residue count matches between implementations."""
        # Run original to get mask
        orig_log_p, orig_mask, _ = run_original_unconditional_probs(
            runner, pdb_path, seed=42
        )

        # Run new implementation
        result = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )

        # Count should match
        expected_count = int(orig_mask.sum())
        assert len(result.residue_info) == expected_count
        assert result.log_probs.shape[0] == expected_count

    def test_log_probs_shape_matches(self, runner, pdb_path):
        """Test that log_probs shape is correct."""
        result = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=True,
            seed=42,
        )

        # Should have 21 amino acid columns
        assert result.log_probs.shape[1] == 21

    def test_deterministic_with_same_seed(self, runner, pdb_path):
        """Test that results are deterministic with same seed."""
        result1 = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=False,
            seed=12345,
        )
        result2 = runner.compute_probs(
            pdb_path=pdb_path,
            unconditional=False,
            seed=12345,
        )

        np.testing.assert_array_equal(
            result1.log_probs,
            result2.log_probs,
            err_msg="Results not deterministic with same seed",
        )

"""Inference runner for ProteinMPNN."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from proteinmpnn.data.single_state import SingleStateDesignInput
from proteinmpnn.data.structure.dataset import StructureDatasetPDB
from proteinmpnn.inference.results import DesignResult, NativeSequence, SequenceResult
from proteinmpnn.inference.transform import transform_inputs
from proteinmpnn.model.proteinmpnn import ProteinMPNN, _S_to_seq, _scores, tied_featurize
from proteinmpnn.utils.constants import AA, HIDDEN_DIM, NUM_LAYERS, WEIGHTS_PATH

if TYPE_CHECKING:
    from proteinmpnn.data.config import SingleStateConfig

# Type alias for model names
ModelName = Literal[
    "v_48_002",
    "v_48_010",
    "v_48_020",
    "v_48_030",
    "ca_48_002",
    "ca_48_010",
    "ca_48_020",
    "s_48_002",
    "s_48_010",
    "s_48_020",
    "s_48_030",
]


class InferenceRunner:
    """High-level API for running ProteinMPNN inference.

    This class handles model loading and provides methods for designing
    sequences for protein structures.

    Args:
        model_name: Name of the ProteinMPNN model to use.
        backbone_noise: Standard deviation of Gaussian noise to add to backbone.
        device: PyTorch device to use. If None, auto-selects cuda > mps > cpu.

    Example:
        >>> runner = InferenceRunner()
        >>> result = runner.design_single(
        ...     pdb_path="6MRR.pdb",
        ...     designable_res="A1-A68",
        ...     num_sequences=5
        ... )
        >>> print(result.to_fasta())
    """

    def __init__(
        self,
        model_name: ModelName = "v_48_020",
        backbone_noise: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.backbone_noise = backbone_noise

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> ProteinMPNN:
        """Load the ProteinMPNN model from checkpoint."""
        ckpt_path = WEIGHTS_PATH / f"{self.model_name}.pt"
        if not ckpt_path.exists():
            raise ValueError(
                f"Model weights not found at {ckpt_path}. "
                f"Please ensure the model weights are downloaded."
            )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        num_edges = ckpt["num_edges"]

        model = ProteinMPNN(
            num_letters=21,
            node_features=HIDDEN_DIM,
            edge_features=HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            augment_eps=self.backbone_noise,
            k_neighbors=num_edges,
        )
        model.to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return model

    def design_single(
        self,
        pdb_path: str | Path,
        designable_res: str = "",
        symmetric_res: str = "",
        cluster_center: str = "",
        cluster_radius: float = 10.0,
        num_sequences: int = 1,
        batch_size: int = 1,
        temperatures: list[float] | None = None,
        seed: int | None = None,
        dump_probs: bool = False,
    ) -> DesignResult:
        """Design sequences for a single protein structure.

        Args:
            pdb_path: Path to the input PDB file.
            designable_res: Comma-separated residue specifications
                (e.g., "A10,A12-A15").
            symmetric_res: Comma-separated symmetry constraints
                (e.g., "A10:B10,A11:B11").
            cluster_center: Center residue(s) for cluster-based design.
            cluster_radius: Radius in Angstroms for cluster-based selection.
            num_sequences: Total number of sequences to generate.
            batch_size: Batch size for generation.
            temperatures: List of sampling temperatures. Defaults to [0.1].
            seed: Random seed for reproducibility.
            dump_probs: Whether to return probability matrices.

        Returns:
            DesignResult containing native and designed sequences.
        """
        pdb_path = Path(pdb_path)

        # Create design input specification
        design_input = SingleStateDesignInput(
            pdb_path=pdb_path,
            designable_res=designable_res,
            default_design_setting="all",
            symmetric_res=symmetric_res,
            cluster_center=cluster_center,
            cluster_radius=cluster_radius,
        )
        config = design_input.to_config()

        return self.design_from_config(
            pdb_path=pdb_path,
            config=config,
            num_sequences=num_sequences,
            batch_size=batch_size,
            temperatures=temperatures,
            seed=seed,
            dump_probs=dump_probs,
        )

    def design_from_config(
        self,
        pdb_path: str | Path,
        config: SingleStateConfig,
        num_sequences: int = 1,
        batch_size: int = 1,
        temperatures: list[float] | None = None,
        seed: int | None = None,
        dump_probs: bool = False,
    ) -> DesignResult:
        """Design sequences using a pre-built config.

        Args:
            pdb_path: Path to the input PDB file.
            config: SingleStateConfig with design specifications.
            num_sequences: Total number of sequences to generate.
            batch_size: Batch size for generation.
            temperatures: List of sampling temperatures. Defaults to [0.1].
            seed: Random seed for reproducibility.
            dump_probs: Whether to return probability matrices.

        Returns:
            DesignResult containing native and designed sequences.
        """
        pdb_path = Path(pdb_path)
        if temperatures is None:
            temperatures = [0.1]

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Load protein structure
        pdb_dataset = StructureDatasetPDB.from_pdb_dir(pdb_path.parent)

        # Find the protein matching our PDB
        protein = None
        for p in pdb_dataset:
            if p["name"] == pdb_path.stem:
                protein = p
                break

        if protein is None:
            raise ValueError(f"Could not find protein {pdb_path.stem} in dataset")

        # Run inference
        return self._run_inference(
            protein=protein,
            config=config,
            num_sequences=num_sequences,
            batch_size=batch_size,
            temperatures=temperatures,
            dump_probs=dump_probs,
        )

    def _run_inference(
        self,
        protein: dict,
        config: SingleStateConfig,
        num_sequences: int,
        batch_size: int,
        temperatures: list[float],
        dump_probs: bool,
    ) -> DesignResult:
        """Run the actual inference loop.

        This is the core inference logic ported from run_protein_mpnn_func().
        """
        num_batches = num_sequences // batch_size
        batch_copies = batch_size

        # Default omit_AAs: only omit 'X'
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        omit_aas_np = np.array([aa == "X" for aa in alphabet]).astype(np.float32)
        bias_aas_np = np.zeros(len(alphabet), dtype=np.float32)

        # Transform config into model input dictionaries
        (
            chain_id_dict,
            fixed_positions_dict,
            pssm_dict,
            omit_aa_dict,
            _,
            tied_positions_dict,
            bias_by_res_dict,
        ) = transform_inputs(config, protein)

        # Create batch
        batch_clones = [copy.deepcopy(protein) for _ in range(batch_copies)]

        # Featurize
        (
            X,
            S,
            mask,
            _,
            chain_M,
            chain_encoding_all,
            _,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            _,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = tied_featurize(
            batch_clones,
            self.device,
            chain_id_dict,
            fixed_positions_dict,
            omit_aa_dict,
            tied_positions_dict,
            pssm_dict,
            bias_by_res_dict,
        )

        pssm_threshold = 0.0
        pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()
        name_ = batch_clones[0]["name"]

        # Run inference
        with torch.no_grad():
            # Compute native score
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = self.model(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            native_scores, _ = _scores(S, log_probs, mask_for_loss)
            native_score = native_scores.cpu().data.numpy().mean()

            # Storage for results
            sequences: list[SequenceResult] = []
            probs_list: list[np.ndarray] = []

            # Generate sequences
            for temp in temperatures:
                for j in range(num_batches):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)

                    # PSSM flags
                    pssm_multi = 0.0
                    pssm_log_odds_flag = False
                    pssm_bias_flag = False

                    # Sample using appropriate method
                    if tied_positions_dict is None:
                        sample_dict = self.model.sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=temp,
                            omit_AAs_np=omit_aas_np,
                            bias_AAs_np=bias_aas_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=pssm_log_odds_flag,
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=pssm_bias_flag,
                            bias_by_res=bias_by_res_all,
                        )
                    else:
                        sample_dict = self.model.tied_sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=temp,
                            omit_AAs_np=omit_aas_np,
                            bias_AAs_np=bias_aas_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=pssm_log_odds_flag,
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=pssm_bias_flag,
                            tied_pos=tied_pos_list_of_lists_list[0],
                            tied_beta=tied_beta,
                            bias_by_res=bias_by_res_all,
                        )

                    S_sample = sample_dict["S"]

                    # Score the sample
                    log_probs = self.model(
                        X,
                        S_sample,
                        mask,
                        chain_M * chain_M_pos,
                        residue_idx,
                        chain_encoding_all,
                        randn_2,
                        use_input_decoding_order=True,
                        decoding_order=sample_dict["decoding_order"],
                    )
                    mask_for_loss = mask * chain_M * chain_M_pos
                    scores, _ = _scores(S_sample, log_probs, mask_for_loss)
                    scores_np = scores.cpu().data.numpy()

                    if dump_probs:
                        probs_list.append(sample_dict["probs"].cpu().data.numpy())

                    # Process each sample in batch
                    for b_ix in range(batch_copies):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]

                        # Calculate sequence recovery
                        seq_recovery_rate = torch.sum(
                            torch.sum(
                                torch.nn.functional.one_hot(S[b_ix], 21)
                                * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                                axis=-1,
                            )
                            * mask_for_loss[b_ix]
                        ) / torch.sum(mask_for_loss[b_ix])

                        # Convert to sequence string
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores_np[b_ix]

                        # Reorder chains and add separators
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(
                            list(np.array(list_of_AAs)[np.argsort(masked_list)])
                        )
                        l0 = 0
                        for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(masked_list)]
                        )[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + "/" + seq[l0:]
                            l0 += 1

                        sequences.append(
                            SequenceResult(
                                sequence=seq,
                                score=float(score),
                                seq_recovery=float(
                                    seq_recovery_rate.detach().cpu().numpy()
                                ),
                                temperature=temp,
                                sample_index=b_ix,
                            )
                        )

            # Build native sequence string
            native_seq = _S_to_seq(S[0], chain_M[0])
            masked_chain_length_list = masked_chain_length_list_list[0]
            masked_list = masked_list_list[0]

            start = 0
            end = 0
            list_of_AAs = []
            for mask_l in masked_chain_length_list:
                end += mask_l
                list_of_AAs.append(native_seq[start:end])
                start = end

            native_seq = "".join(
                list(np.array(list_of_AAs)[np.argsort(masked_list)])
            )
            l0 = 0
            for mc_length in list(
                np.array(masked_chain_length_list)[np.argsort(masked_list)]
            )[:-1]:
                l0 += mc_length
                native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                l0 += 1

            # Get chain info
            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
            print_masked_chains = [
                masked_list_list[0][i] for i in sorted_masked_chain_letters
            ]
            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
            print_visible_chains = [
                visible_list_list[0][i] for i in sorted_visible_chain_letters
            ]

            native = NativeSequence(
                name=name_,
                sequence=native_seq,
                score=float(native_score),
                fixed_chains=print_visible_chains,
                designed_chains=print_masked_chains,
                model_name=self.model_name,
            )

            # Average probs if requested
            probs = None
            if dump_probs and probs_list:
                probs = np.squeeze(np.mean(np.stack(probs_list), axis=0), axis=0)

            return DesignResult(
                protein_name=name_,
                native=native,
                sequences=sequences,
                probs=probs,
            )

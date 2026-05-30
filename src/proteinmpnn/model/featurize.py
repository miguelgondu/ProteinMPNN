"""Featurization functions for ProteinMPNN.

This module contains the tied_featurize function which converts protein
structure data into tensor representations for the ProteinMPNN model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def tied_featurize(
    batch: list[dict[str, Any]],
    device: torch.device | str,
    chain_dict: dict[str, tuple[list[str], list[str]]] | None,
    fixed_position_dict: dict[str, dict[str, list[int]]] | None = None,
    omit_AA_dict: dict[str, dict[str, list[list[int | str]]]] | None = None,
    tied_positions_dict: dict[str, list[dict[str, list[int]]]] | None = None,
    pssm_dict: dict[str, dict[str, dict[str, Any]]] | None = None,
    bias_by_res_dict: dict[str, np.ndarray] | None = None,
) -> tuple[
    torch.Tensor,  # X
    torch.Tensor,  # S
    torch.Tensor,  # mask
    np.ndarray,  # lengths
    torch.Tensor,  # chain_M
    torch.Tensor,  # chain_encoding_all
    list[list[str]],  # letter_list_list
    list[list[str]],  # visible_list_list
    list[list[str]],  # masked_list_list
    list[list[int]],  # masked_chain_length_list_list
    torch.Tensor,  # chain_M_pos
    torch.Tensor,  # omit_AA_mask
    torch.Tensor,  # residue_idx
    torch.Tensor,  # dihedral_mask
    list[list[list[int]]],  # tied_pos_list_of_lists_list
    torch.Tensor,  # pssm_coef_all
    torch.Tensor,  # pssm_bias_all
    torch.Tensor,  # pssm_log_odds_all
    torch.Tensor,  # bias_by_res_all
    torch.Tensor,  # tied_beta
]:
    """Pack and pad batch of protein structures into torch tensors.

    This function converts protein structure dictionaries into the tensor
    format required by the ProteinMPNN model, handling batching, padding,
    masking, and various design constraints.

    Args:
        batch: List of protein dictionaries from StructureDatasetPDB.
        device: PyTorch device for output tensors.
        chain_dict: Mapping from protein name to (masked_chains, visible_chains).
            If None, all chains are treated as masked (designable).
        fixed_position_dict: Mapping from protein name -> chain -> list of
            fixed position indices (1-indexed).
        omit_AA_dict: Mapping from protein name -> chain -> list of
            [resid, omit_AAs] pairs specifying amino acids to omit.
        tied_positions_dict: Mapping from protein name -> list of tied position
            dictionaries for symmetric design.
        pssm_dict: PSSM-based biasing information.
        bias_by_res_dict: Per-residue amino acid biases.

    Returns:
        A 20-tuple of tensors and lists for model input:
        - X: Backbone coordinates [B, L, 4, 3]
        - S: Sequence indices [B, L]
        - mask: Valid position mask [B, L]
        - lengths: Sequence lengths [B]
        - chain_M: Chain mask (1 for designable) [B, L]
        - chain_encoding_all: Chain IDs [B, L]
        - letter_list_list: Chain letters per batch
        - visible_list_list: Visible (fixed) chains per batch
        - masked_list_list: Masked (designable) chains per batch
        - masked_chain_length_list_list: Lengths of masked chains
        - chain_M_pos: Position-level design mask [B, L]
        - omit_AA_mask: Amino acids to omit [B, L, 21]
        - residue_idx: Residue indices [B, L]
        - dihedral_mask: Dihedral angle mask [B, L, 3]
        - tied_pos_list_of_lists_list: Tied positions for symmetric design
        - pssm_coef_all: PSSM coefficients [B, L]
        - pssm_bias_all: PSSM biases [B, L, 21]
        - pssm_log_odds_all: PSSM log odds [B, L, 21]
        - bias_by_res_all: Per-residue biases [B, L, 21]
        - tied_beta: Tied position weights [L]
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros(
        [B, L_max], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[
                b["name"]
            ]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == "seq_chain_"]
            visible_chains = []
        all_chains = masked_chains + visible_chains
    for i, b in enumerate(batch):
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        chain_coords[c]
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                x_chain = np.stack(
                    [
                        chain_coords[c]
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict is not None:
                    fixed_pos_list = fixed_position_dict[b["name"]][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict is not None:
                    for item in omit_AA_dict[b["name"]][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array(
                            [
                                np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                                for AA in item[1]
                            ]
                        )
                        idx_ = np.array([[idx_AA, b] for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict and pssm_dict[b["name"]][letter]:
                    pssm_coef = pssm_dict[b["name"]][letter]["pssm_coef"]
                    pssm_bias = pssm_dict[b["name"]][letter]["pssm_bias"]
                    pssm_log_odds = pssm_dict[b["name"]][letter]["pssm_log_odds"]
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict is not None:
            tied_pos_list = tied_positions_dict[b["name"]]
            if tied_pos_list:
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[
                            np.argwhere(letter_list_np == k)[0][0]
                        ]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(
                                    start_idx + v[0][v_count] - 1
                                )  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(
                                    start_idx + v_ - 1
                                )  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(
            fixed_position_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(
            pssm_coef_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(
            pssm_bias_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(
            pssm_log_odds_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(
            bias_by_res_list, 0
        )  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], "constant", constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(
            np.concatenate(omit_AA_mask_list, 0),
            [[0, L_max - l]],
            "constant",
            constant_values=(0.0,),
        )
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(
            pssm_coef_, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        pssm_bias_pad = np.pad(
            pssm_bias_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        pssm_log_odds_pad = np.pad(
            pssm_log_odds_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(
            bias_by_res_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(
        dtype=torch.float32, device=device
    )
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(
        dtype=torch.float32, device=device
    )
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float32, device=device
    )

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(
        dtype=torch.float32, device=device
    )
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(
        dtype=torch.float32, device=device
    )
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    return (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef_all,
        pssm_bias_all,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
    )

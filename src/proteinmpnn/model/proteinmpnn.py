"""ProteinMPNN model for protein sequence design.

This module contains the main ProteinMPNN neural network model which
predicts amino acid sequences given protein backbone structures.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import proteinmpnn.sample.metropolis as metropolis_sample
from proteinmpnn.model.layers import DecLayer, EncLayer, ProteinFeatures
from proteinmpnn.model.losses import scores
from proteinmpnn.model.utils import (
    cat_neighbors_nodes,
    gather_nodes,
)


class ProteinMPNN(nn.Module):
    """ProteinMPNN model for structure-conditioned sequence design.

    This model uses a message-passing neural network architecture to
    predict amino acid sequences that are compatible with a given
    protein backbone structure.

    Args:
        num_letters: Size of the output vocabulary (typically 21 for amino acids).
        node_features: Dimension of node feature embeddings.
        edge_features: Dimension of edge feature embeddings.
        hidden_dim: Hidden dimension for encoder/decoder layers.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        vocab: Vocabulary size for sequence embeddings.
        k_neighbors: Number of nearest neighbors in the protein graph.
        augment_eps: Standard deviation for coordinate augmentation noise.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_letters: int,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 64,
        augment_eps: float = 0.05,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor,
        use_input_decoding_order: bool = False,
        decoding_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing log probabilities for each position.

        Args:
            X: Backbone coordinates [B, L, 4, 3].
            S: Sequence indices [B, L].
            mask: Valid position mask [B, L].
            chain_M: Chain mask (1 for designable) [B, L].
            residue_idx: Residue indices [B, L].
            chain_encoding_all: Chain IDs [B, L].
            randn: Random tensor for decoding order [B, L].
            use_input_decoding_order: Whether to use provided decoding_order.
            decoding_order: Optional pre-specified decoding order [B, L].

        Returns:
            Log probabilities of shape [B, L, vocab].
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions
        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and
            # higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)

    def sample(
        self,
        X,
        randn,
        S_true,
        chain_mask,
        chain_encoding_all,
        residue_idx,
        mask=None,
        temperature=1.0,
        omit_AAs_np=None,
        bias_AAs_np=None,
        chain_M_pos=None,
        omit_AA_mask=None,
        pssm_coef=None,
        pssm_bias=None,
        pssm_multi=None,
        pssm_log_odds_flag=None,
        pssm_log_odds_mask=None,
        pssm_bias_flag=None,
        bias_by_res=None,
        invert_probs=False,
    ):
        """Sample sequences from the model.

        Args:
            X: Backbone coordinates [B, L, 4, 3].
            randn: Random tensor for decoding order [B, L].
            S_true: Ground truth sequence [B, L].
            chain_mask: Chain mask (1 for designable) [B, L].
            chain_encoding_all: Chain IDs [B, L].
            residue_idx: Residue indices [B, L].
            mask: Valid position mask [B, L].
            temperature: Sampling temperature.
            omit_AAs_np: Amino acids to omit globally.
            bias_AAs_np: Global amino acid biases.
            chain_M_pos: Position-level design mask [B, L].
            omit_AA_mask: Per-position amino acid omission [B, L, 21].
            pssm_coef: PSSM coefficients.
            pssm_bias: PSSM biases.
            pssm_multi: PSSM multiplier.
            pssm_log_odds_flag: Whether to use PSSM log odds.
            pssm_log_odds_mask: PSSM log odds mask.
            pssm_bias_flag: Whether to use PSSM bias.
            bias_by_res: Per-residue biases [B, L, 21].
            invert_probs: Whether to invert probabilities.

        Returns:
            Dictionary with 'S' (sampled sequences), 'probs', and 'decoding_order'.
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = (
            chain_mask * chain_M_pos * mask
        )  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and
        # higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float32
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        # chain_mask_combined = chain_mask*chain_M_pos
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]
            bias_by_res_gathered = torch.gather(
                bias_by_res, 1, t[:, None, None].repeat(1, 1, 21)
            )[
                :, 0, :
            ]  # [B, 21]
            if (chain_mask_gathered == 0).all():
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                )
                h_E_t = torch.gather(
                    h_E,
                    1,
                    t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]),
                )
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw,
                    1,
                    t[:, None, None, None].repeat(
                        1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                    ),
                )
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                    )
                    h_ESV_t = (
                        torch.gather(
                            mask_bw,
                            1,
                            t[:, None, None, None].repeat(
                                1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                            ),
                        )
                        * h_ESV_decoder_t
                        + h_EXV_encoder_t
                    )
                    h_V_stack[l + 1].scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t),
                    )

                # Sampling step
                h_V_t = torch.gather(
                    h_V_stack[-1],
                    1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                )[:, 0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(
                    logits
                    - constant[None, :] * 1e8
                    + constant_bias[None, :] / temperature
                    + bias_by_res_gathered / temperature,
                    dim=-1,
                )
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(
                        pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1])
                    )[:, 0]
                    probs = (
                        1 - pssm_multi * pssm_coef_gathered[:, None]
                    ) * probs + pssm_multi * pssm_coef_gathered[
                        :, None
                    ] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(
                        pssm_log_odds_mask,
                        1,
                        t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1]),
                    )[
                        :, 0
                    ]  # [B, 21]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                if invert_probs:
                    print(constant.shape)
                    probs = (1.0 / (probs + 1e-12)) * (1.0 - constant[None, :])
                    probs = probs / torch.sum(probs)
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(
                        omit_AA_mask,
                        1,
                        t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]),
                    )[
                        :, 0
                    ]  # [B, 21]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True
                    )  # [B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(
                    1,
                    t[:, None, None].repeat(1, 1, 21),
                    (
                        chain_mask_gathered[
                            :,
                            :,
                            None,
                        ]
                        * probs[:, None, :]
                    ).float(),
                )
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (
                S_t * chain_mask_gathered
                + S_true_gathered * (1.0 - chain_mask_gathered)
            ).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
            S.scatter_(1, t[:, None], S_t)
        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def tied_sample(
        self,
        X,
        randn,
        S_true,
        chain_mask,
        chain_encoding_all,
        residue_idx,
        mask=None,
        temperature=1.0,
        omit_AAs_np=None,
        bias_AAs_np=None,
        chain_M_pos=None,
        omit_AA_mask=None,
        pssm_coef=None,
        pssm_bias=None,
        pssm_multi=None,
        pssm_log_odds_flag=None,
        pssm_log_odds_mask=None,
        pssm_bias_flag=None,
        tied_pos=None,
        tied_beta=None,
        bias_by_res=None,
        invert_probs=False,
        bidir=False,
        bidir_table_dir=None,
    ):
        """Sample sequences with tied (symmetric) positions.

        Similar to sample() but handles tied positions that should have
        the same amino acid.

        Args:
            tied_pos: List of lists of tied position indices.
            tied_beta: Weights for tied positions.
            bidir: Whether to use bidirectional sampling.
            bidir_table_dir: Directory containing bidirectional lookup table.
            (Other args same as sample())

        Returns:
            Dictionary with 'S', 'probs', and 'decoding_order'.
        """
        device = X.device
        if bidir:
            if not isinstance(bidir_table_dir, (str, Path)):
                raise ValueError("Asked for bidirectional, but no table dir was given.")

            if isinstance(bidir_table_dir, str):
                bidir_table_dir = Path(bidir_table_dir)

            bidir_filter = torch.load(
                bidir_table_dir / "bidir_table.pt", map_location=device
            )
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = (
            chain_mask * chain_M_pos * mask
        )  # update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  # [numbers will be smaller for places where chain_M = 0.0 and
        # higher for places where chain_M = 1.0]
        new_decoding_order = []
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(
            list(itertools.chain(*new_decoding_order)), device=device
        )[
            None,
        ].repeat(
            X.shape[0], 1
        )
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        all_probs = torch.zeros(
            (N_batch, N_nodes, 21), device=device, dtype=torch.float32
        )
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask is not None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (chain_mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                E_idx_t = E_idx[:, t : t + 1, :]
                h_E_t = h_E[:, t : t + 1, :, :]
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = h_EXV_encoder_fw[:, t : t + 1, :, :]
                mask_t = mask[:, t : t + 1]
                for l, layer in enumerate(self.decoder_layers):
                    h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = h_V_stack[l][:, t : t + 1, :]
                    h_ESV_t = (
                        mask_bw[:, t : t + 1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                    )
                    h_V_stack[l + 1][:, t, :] = layer(
                        h_V_t, h_ESV_t, mask_V=mask_t
                    ).squeeze(1)
                h_V_t = h_V_stack[-1][:, t, :]
                logit_list.append((self.W_out(h_V_t) / temperature) / len(t_list))
                logits += tied_beta[t] * (self.W_out(h_V_t) / temperature) / len(t_list)
            if done_flag:
                pass
            else:
                # bidirectional coding using modified tied_sample
                if bidir and len(logit_list) == 2 and bidir_table_dir:
                    b1, b2 = t_list  # need to keep the positions for bias terms
                    # calculate all combinations of logits by addition (less
                    # harsh than multiplying)
                    a, b = torch.flatten(logit_list[0]), torch.flatten(logit_list[1])
                    probs = a.unsqueeze(1) + b.unsqueeze(0)

                    # need to square all per-AA bias terms for modifying 2D prob matrix
                    square_constant = torch.flatten(constant[None, :])
                    square_constant = torch.outer(square_constant, square_constant)
                    square_constant_bias = torch.flatten(constant_bias[None, :])
                    square_constant_bias = torch.outer(
                        square_constant_bias, square_constant_bias
                    )
                    square_bias_by_res_gathered = torch.outer(
                        torch.flatten(bias_by_res[:, 0, :]),
                        torch.flatten(bias_by_res[:, 1, :]),
                    )

                    probs = (
                        probs
                        - square_constant[None, :] * 1e8
                        + square_constant_bias[None, :] / temperature
                        + square_bias_by_res_gathered / temperature
                    )
                    probs = F.softmax(probs, dim=-1)
                    # for all bias terms, they again need to be squared into 2D - this
                    # is blunt but works
                    if pssm_bias_flag:
                        pssm_coef_gathered = torch.outer(
                            pssm_coef[:, b1], pssm_coef[:, b2]
                        )
                        pssm_bias_gathered = torch.outer(
                            pssm_bias[:, b1], pssm_bias[:, b2]
                        )
                        probs = (
                            1 - pssm_multi * pssm_coef_gathered[:, None]
                        ) * probs + pssm_multi * pssm_coef_gathered[
                            :, None
                        ] * pssm_bias_gathered
                    if pssm_log_odds_flag:
                        pssm_log_odds_mask_gathered = torch.outer(
                            pssm_log_odds_mask[:, b1], pssm_log_odds_mask[:, b2]
                        )
                        probs_masked = probs * pssm_log_odds_mask_gathered
                        probs_masked += probs * 0.001
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]
                    if omit_AA_mask_flag:
                        omit_AA_mask_gathered = 1 - torch.outer(
                            torch.flatten(1 - omit_AA_mask[:, b1]),
                            torch.flatten(1 - omit_AA_mask[:, b2]),
                        )
                        probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]

                    probs = torch.nan_to_num(
                        probs, nan=0.0
                    )  # sometimes omit_AA_mask division can make NaNs
                    probs = (
                        probs * bidir_filter
                    )  # filter by binary table of allowed bidirectional AA combinations
                    # need to flatten probs for easier softmax and sampling operations
                    flat_probs = torch.flatten(probs)  # [441]

                    # sampling options from the multinomial distribution based
                    # on probs weighting
                    probs = probs.squeeze()
                    p_shape = probs.shape
                    try:
                        S_t_repeat = torch.multinomial(flat_probs, 1).squeeze(-1)
                    except RuntimeError:
                        print(
                            "*** Invalid multinomial distribution (sum of probabilities"
                            " <= 0). This means there is NO valid bidirect. AA combo"
                            " to choose from - check your AA constraints  ***"
                        )
                        quit()
                    # extract idx of each AA from prob sampling by reverse-engineering
                    # the flatten operation
                    prob_b = S_t_repeat % p_shape[0]
                    prob_a = (S_t_repeat - prob_b) / p_shape[0]
                    combo_prob = probs[prob_a.long(), prob_b.long()]

                    # need to handle each half of the tied pair separately, since
                    # they can be different AAs
                    for t, p_idx in zip(t_list, [prob_a, prob_b]):
                        S_t_repeat = (
                            chain_mask[:, t] * p_idx
                            + (1 - chain_mask[:, t]) * S_true[:, t]
                        ).long()
                        h_S[:, t, :] = self.W_s(S_t_repeat)
                        S[:, t] = S_t_repeat
                        all_probs[:, t, :] = combo_prob.float()

                else:  # default MPNN tied decoding
                    bias_by_res_gathered = bias_by_res[:, t, :]  # [B, 21]
                    probs = F.softmax(
                        logits
                        - constant[None, :] * 1e8
                        + constant_bias[None, :] / temperature
                        + bias_by_res_gathered / temperature,
                        dim=-1,
                    )
                    if pssm_bias_flag:
                        pssm_coef_gathered = pssm_coef[:, t]
                        pssm_bias_gathered = pssm_bias[:, t]
                        probs = (
                            1 - pssm_multi * pssm_coef_gathered[:, None]
                        ) * probs + pssm_multi * pssm_coef_gathered[
                            :, None
                        ] * pssm_bias_gathered
                    if pssm_log_odds_flag:
                        pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                        probs_masked = probs * pssm_log_odds_mask_gathered
                        probs_masked += probs * 0.001
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]
                    if invert_probs:
                        probs = (1.0 / (probs + 1e-12)) * (1.0 - constant[None, :])
                        probs = probs / torch.sum(probs)
                    if omit_AA_mask_flag:
                        omit_AA_mask_gathered = omit_AA_mask[:, t]
                        probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]
                    S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t_repeat)
                        S[:, t] = S_t_repeat
                        all_probs[:, t, :] = probs.float()
        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def mcmc_sample(self, X, mask, residue_idx, chain_encoding_all, temperature=1.0):
        """Bidirectional sequence sampler using MCMC based sampler."""

        # 1. temperature-dependent unconditional prob sampler
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # NOTE: this is the key variable that makes it a one-shot decoder - all
        # seq info is masked out
        order_mask_backward = torch.zeros(
            [X.shape[0], X.shape[1], X.shape[1]], device=device
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = (
            self.W_out(h_V) / temperature + 1e-20
        )  # scale raw logits by temperature (lower = more spiky)
        probs = F.softmax(
            logits, dim=-1
        )  # [1, L, 21] array of probabilities for both chains
        prob_chain_list = []
        for chain_idx in torch.unique(chain_encoding_all):
            prob_chain_list.append(probs[chain_encoding_all == chain_idx])

        # prob_chain_list is a list of tensors of length number_of_chains (e.g., 2)
        # each tensor is a seti of probabilities of shape [L, 21] where L is
        # the length of that chain
        (
            out_list1,
            out_list2,
            score1,
            score2,
            final_AAs1,
            final_AAs2,
            DNA_seq1,
            DNA_seq2,
        ) = metropolis_sample.na_sample(prob_chain_list[0], prob_chain_list[1])

        # reshape probabilities to include batch dimension
        B = X.shape[0]
        assert B == 1, "Batch size > 1 not tested for MCMC sampling."
        probs1 = out_list1.expand(B, -1, -1)
        probs2 = out_list2.expand(B, -1, -1)
        # concatenate the two strands
        all_probs = torch.cat([probs1, probs2], dim=1)

        # reduce one-hot encoding to integer index
        S = torch.argmax(all_probs, dim=-1)

        # Make placeholder decoding order
        decoding_order = (
            torch.arange(0, S.shape[1], device=S.device)
            .unsqueeze(0)
            .repeat(S.shape[0], 1)
        )

        # NOTE: to return other scores, just add them to this output dict and parse
        # them later with EvoPro
        return {"S": S, "probs": all_probs, "decoding_order": decoding_order}

    def conditional_probs(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        randn,
        backbone_only=False,
    ):
        """Graph-conditioned sequence model"""
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions

        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros(
            [X.shape[0], chain_M.shape[1], 21], device=device
        ).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1], device=device).float()
                order_mask[idx] = 0.0
            else:
                order_mask = torch.zeros(chain_M.shape[1], device=device).float()
                order_mask[idx] = 1.0
            decoding_order = torch.argsort(
                (order_mask[None,] + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and
            # higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """Graph-conditioned sequence model"""
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros(
            [X.shape[0], X.shape[1], X.shape[1]], device=device
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        return F.log_softmax(logits, dim=-1)

    def pairwise_sample(
        self,
        X,
        randn,
        S_true,
        chain_mask,
        chain_encoding_all,
        residue_idx,
        mask=None,
        temperature=1.0,
        omit_AAs_np=None,
        bias_AAs_np=None,
        chain_M_pos=None,
        omit_AA_mask=None,
        pssm_coef=None,
        pssm_bias=None,
        pssm_multi=None,
        pssm_log_odds_flag=None,
        pssm_log_odds_mask=None,
        pssm_bias_flag=None,
        bias_by_res=None,
        invert_probs=False,
    ):
        """
        Samples all possible permutations of two position lists and keeps
        the best combination of mutations.
        EXPERIMENTAL.
        """

        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = (
            chain_mask * chain_M_pos * mask
        )  # update chain_M to include missing regions
        true_chain_mask = chain_mask.clone()
        # get two sets of mutated positions from chain_mask and chain_encoding
        designable_positions = torch.where(chain_mask != 0.0)[-1]
        designable_chains = torch.gather(
            chain_encoding_all,
            dim=-1,
            index=torch.unsqueeze(designable_positions, dim=0),
        )
        unique_chains = torch.unique(designable_chains)
        assert unique_chains.size()[0] == 2

        positions_1 = torch.where(designable_chains == unique_chains[0])[-1]
        positions_1 = designable_positions[positions_1]
        positions_2 = torch.where(designable_chains == unique_chains[1])[-1]
        positions_2 = designable_positions[positions_2]

        # compile all N possible permutations of the two lists [N, 2]
        # for lists of size A and B, N = A x B
        prod = torch.cat(
            [
                torch.cartesian_prod(positions_1, positions_2),
                torch.cartesian_prod(positions_2, positions_1),
            ],
            dim=0,
        )
        output_dict_list = []
        log_probs_list = []

        print(f"{prod.shape[0]} pair permutations to sample!")
        # iterate through each permutation and run decoding on only these two positions
        for p_ix in range(prod.shape[0]):
            perm = prod[p_ix, :]  # [2,]

            chain_mask_perm = torch.zeros_like(chain_mask, dtype=torch.float32)
            chain_mask_perm[:, perm[0]] = 1.0
            chain_mask_perm[:, perm[1]] = 2.0

            decoding_order = torch.argsort(
                (chain_mask_perm + 0.0001) * (torch.abs(randn))
            )  # [numbers will be smaller for places where chain_M = 0.0 and
            # higher for places where chain_M = 1.0]
            chain_mask_perm[:, perm[1]] = 1.0
            chain_mask = chain_mask_perm

            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1.0 - mask_attend)

            N_batch, N_nodes = X.size(0), X.size(1)
            log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
            all_probs = torch.zeros(
                (N_batch, N_nodes, 21), device=device, dtype=torch.float32
            )
            h_S = torch.zeros_like(h_V, device=device)
            S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
            h_V_stack = [h_V] + [
                torch.zeros_like(h_V, device=device)
                for _ in range(len(self.decoder_layers))
            ]
            constant = torch.tensor(omit_AAs_np, device=device)
            constant_bias = torch.tensor(bias_AAs_np, device=device)
            # chain_mask_combined = chain_mask*chain_M_pos
            omit_AA_mask_flag = omit_AA_mask is not None

            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for t_ in range(N_nodes):
                t = decoding_order[:, t_]  # [B]
                chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]
                bias_by_res_gathered = torch.gather(
                    bias_by_res, 1, t[:, None, None].repeat(1, 1, 21)
                )[
                    :, 0, :
                ]  # [B, 21]
                if (
                    chain_mask_gathered == 0
                ).all():  # if position is fixed, just fill in sequence
                    S_t = torch.gather(S_true, 1, t[:, None])
                else:
                    # Hidden layers
                    E_idx_t = torch.gather(
                        E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
                    )
                    h_E_t = torch.gather(
                        h_E,
                        1,
                        t[:, None, None, None].repeat(
                            1, 1, h_E.shape[-2], h_E.shape[-1]
                        ),
                    )
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = torch.gather(
                        h_EXV_encoder_fw,
                        1,
                        t[:, None, None, None].repeat(
                            1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]
                        ),
                    )
                    mask_t = torch.gather(mask, 1, t[:, None])
                    for l, layer in enumerate(self.decoder_layers):
                        # Updated relational features for future states
                        h_ESV_decoder_t = cat_neighbors_nodes(
                            h_V_stack[l], h_ES_t, E_idx_t
                        )
                        h_V_t = torch.gather(
                            h_V_stack[l],
                            1,
                            t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]),
                        )
                        h_ESV_t = (
                            torch.gather(
                                mask_bw,
                                1,
                                t[:, None, None, None].repeat(
                                    1, 1, mask_bw.shape[-2], mask_bw.shape[-1]
                                ),
                            )
                            * h_ESV_decoder_t
                            + h_EXV_encoder_t
                        )
                        h_V_stack[l + 1].scatter_(
                            1,
                            t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                            layer(h_V_t, h_ESV_t, mask_V=mask_t),
                        )

                    # Sampling step
                    h_V_t = torch.gather(
                        h_V_stack[-1],
                        1,
                        t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]),
                    )[:, 0]
                    logits = self.W_out(h_V_t) / temperature
                    probs = F.softmax(
                        logits
                        - constant[None, :] * 1e8
                        + constant_bias[None, :] / temperature
                        + bias_by_res_gathered / temperature,
                        dim=-1,
                    )
                    if pssm_bias_flag:
                        pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[
                            :, 0
                        ]
                        pssm_bias_gathered = torch.gather(
                            pssm_bias,
                            1,
                            t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]),
                        )[:, 0]
                        probs = (
                            1 - pssm_multi * pssm_coef_gathered[:, None]
                        ) * probs + pssm_multi * pssm_coef_gathered[
                            :, None
                        ] * pssm_bias_gathered
                    if pssm_log_odds_flag:
                        pssm_log_odds_mask_gathered = torch.gather(
                            pssm_log_odds_mask,
                            1,
                            t[:, None, None].repeat(1, 1, pssm_log_odds_mask.shape[-1]),
                        )[
                            :, 0
                        ]  # [B, 21]
                        probs_masked = probs * pssm_log_odds_mask_gathered
                        probs_masked += probs * 0.001
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]
                    if invert_probs:
                        print(constant.shape)
                        probs = (1.0 / (probs + 1e-12)) * (1.0 - constant[None, :])
                        probs = probs / torch.sum(probs)
                    if omit_AA_mask_flag:
                        omit_AA_mask_gathered = torch.gather(
                            omit_AA_mask,
                            1,
                            t[:, None, None].repeat(1, 1, omit_AA_mask.shape[-1]),
                        )[
                            :, 0
                        ]  # [B, 21]
                        probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                        probs = probs_masked / torch.sum(
                            probs_masked, dim=-1, keepdim=True
                        )  # [B, 21]
                    S_t = torch.multinomial(probs, 1)
                    all_probs.scatter_(
                        1,
                        t[:, None, None].repeat(1, 1, 21),
                        (
                            chain_mask_gathered[
                                :,
                                :,
                                None,
                            ]
                            * probs[:, None, :]
                        ).float(),
                    )
                S_true_gathered = torch.gather(S_true, 1, t[:, None])
                S_t = (
                    S_t * chain_mask_gathered
                    + S_true_gathered * (1.0 - chain_mask_gathered)
                ).long()
                temp1 = self.W_s(S_t)
                h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
                S.scatter_(1, t[:, None], S_t)
                # scores = scores.cpu().data.numpy()
            output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
            # obtain score from each sequence by passing FULL sequence
            # through forward fxn
            log_probs = self(
                X,
                S,
                mask,
                chain_mask,
                residue_idx,
                chain_encoding_all,
                randn,
                use_input_decoding_order=True,
                decoding_order=output_dict["decoding_order"],
            )
            # mask_for_loss = mask*chain_mask
            mask_for_loss = mask
            score_vals, scores_per_res = scores(S, log_probs, mask_for_loss)
            output_dict["score"] = score_vals
            output_dict["score_per_res"] = scores_per_res
            log_probs_list.append(log_probs)
            output_dict_list.append(output_dict)
        # compile and pick best score/seq to return
        all_scores = torch.tensor([odl["score"] for odl in output_dict_list])
        best_seq = torch.argmin(all_scores).item()
        output_dict = output_dict_list[best_seq]
        log_probs = log_probs_list[best_seq]
        output_dict["score"] = output_dict["score"].cpu().data.numpy()
        output_dict["score_per_res"] = output_dict["score_per_res"].cpu().data.numpy()
        return output_dict, log_probs, true_chain_mask

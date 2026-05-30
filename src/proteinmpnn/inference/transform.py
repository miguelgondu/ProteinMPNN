"""Transform design specifications into model input dictionaries.

This module converts SingleStateConfig into the dictionary formats
required by tied_featurize().
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from proteinmpnn.data.config import SingleStateConfig


# Amino acid alphabet (matches model order)
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# Pre-defined amino acid groups for MutTo parsing
HYDPHOB = {"omit": set("CDEHKNPQRSTX"), "include": ALPHABET.difference("CDEHKNPQRSTX")}
HYDPHIL = {"omit": set("ACFGILMPVWYX"), "include": ALPHABET.difference("ACFGILMPVWYX")}
POLAR = {"omit": set("AGILMFPWVX"), "include": ALPHABET.difference("AGILMFPWVX")}
NONPOLAR = {"omit": set("RNDCQEHKSTYX"), "include": ALPHABET.difference("RNDCQEHKSTYX")}


def parse_mutate_to(res: dict[str, Any], experimental: bool = False) -> str:
    """Parse MutTo specification and return omitted amino acids.

    Args:
        res: Dictionary with 'MutTo' and 'WTAA' keys.
        experimental: Whether to use the new experimental parsing logic.

    Returns:
        String of amino acids to omit at this position.
    """
    if experimental:
        return _form_omit_aa_list_experimental(res)
    return _form_omit_aa_list_legacy(res)


def _form_omit_aa_list_experimental(res: dict[str, Any]) -> str:
    """Parse MutTo using experimental +/- operator syntax.

    Supports complex expressions like 'hydphob+C-P' (hydrophobic, add C, remove P).

    Args:
        res: Dictionary with 'MutTo' and 'WTAA' keys.

    Returns:
        String of amino acids to omit.
    """
    mutate_keys = ["+" + key for key in res["MutTo"].split("+")]
    mutate_keys = [
        [("-" if item[0] != "+" else "") + item for item in key.split("-")]
        for key in mutate_keys
    ]
    mutate_keys = [elem for item in mutate_keys for elem in item]

    omit_aas: set[str] = set("X")
    for key in mutate_keys:
        if key[0] == "+":
            keyword = key[1:]
            if keyword == "hydphob":
                omit_aas = omit_aas.union(HYDPHOB["omit"])
            elif keyword == "hydphil":
                omit_aas = omit_aas.union(HYDPHIL["omit"])
            elif keyword == "polar":
                omit_aas = omit_aas.union(POLAR["omit"])
            elif keyword == "nonpolar":
                omit_aas = omit_aas.union(NONPOLAR["omit"])
            elif keyword == "all":
                omit_aas = omit_aas.difference(ALPHABET)
            elif keyword == "native":
                omit_aas = omit_aas.difference(set(res["WTAA"]))
            elif set(keyword).issubset(ALPHABET):
                omit_aas = omit_aas.difference(set(keyword))
        elif key[0] == "-":
            keyword = key[1:]
            if keyword == "hydphob":
                omit_aas = omit_aas.difference(HYDPHOB["include"])
            elif keyword == "hydphil":
                omit_aas = omit_aas.difference(HYDPHIL["include"])
            elif keyword == "polar":
                omit_aas = omit_aas.difference(POLAR["include"])
            elif keyword == "nonpolar":
                omit_aas = omit_aas.difference(NONPOLAR["include"])
            elif keyword == "all":
                omit_aas = omit_aas.union(ALPHABET)
            elif keyword == "native":
                omit_aas = omit_aas.union(set(res["WTAA"]))
            elif set(keyword).issubset(ALPHABET):
                omit_aas = omit_aas.union(set(keyword))

    return "".join(sorted(omit_aas))


def _form_omit_aa_list_legacy(res: dict[str, Any]) -> str:
    """Parse MutTo using legacy syntax (for backward compatibility).

    Args:
        res: Dictionary with 'MutTo' and 'WTAA' keys.

    Returns:
        String of amino acids to omit.
    """
    if res["MutTo"] == "all":
        return "X"

    omit_aas: str
    if "hydphob" in res["MutTo"]:
        omit_aas = "CDEHKNPQRSTX"
    elif "hydphil" in res["MutTo"]:
        omit_aas = "ACFGILMPVWYX"
    elif "polar" in res["MutTo"]:
        omit_aas = "AGILMFPWV"
    elif "nonpolar" in res["MutTo"]:
        omit_aas = "RNDCQEHKSTY"
    elif "all" in res["MutTo"] and "-" in res["MutTo"] and "+" not in res["MutTo"]:
        omit_aas = "X"
    elif set(res["MutTo"]).issubset(set("ACDEFGHIKLMNPQRSTVWYX")):
        # Omit everything but what is provided
        omit_aas = "ACDEFGHIKLMNPQRSTVWYX"
        for aa in list(res["MutTo"]):
            if aa in omit_aas:
                omit_aas = list(omit_aas)
                omit_aas.remove(aa)
                omit_aas = "".join(omit_aas)
    else:
        omit_aas = "X"

    if "-" in res["MutTo"]:
        to_omit = res["MutTo"].split("-")[-1].split("+")[0]
        omit_aas += to_omit
    if "+" in res["MutTo"]:
        to_not_omit = res["MutTo"].split("+")[-1].split("-")[0]
        for aa in to_not_omit:
            if aa in omit_aas:
                omit_aas = list(omit_aas)
                omit_aas.remove(aa)
                omit_aas = "".join(omit_aas)

    return omit_aas


def make_fixed_positions_dict(
    config: SingleStateConfig, protein: dict[str, Any]
) -> dict[str, dict[str, list[int]]] | None:
    """Create fixed positions dictionary from design config.

    Fixed positions are those NOT in the designable list.

    Args:
        config: The SingleStateConfig with designable residues.
        protein: Protein dictionary from StructureDatasetPDB.

    Returns:
        Dictionary mapping protein name -> chain -> list of fixed position indices.
        Returns None if all positions are designable.
    """
    # Build design_positions: chain -> list of designable resids
    design_positions: dict[str, list[str]] = {}
    for res in config.designable:
        chain = res.chain
        if chain not in design_positions:
            design_positions[chain] = []
        resid_str = str(res.resid)
        if resid_str not in design_positions[chain]:
            design_positions[chain].append(resid_str)

    # Get all chains with non-empty sequences from the protein
    all_chains = []
    for item in list(protein):
        if item[:10] == "seq_chain_":
            chain_letter = item[10:]
            if protein[item]:  # Only include non-empty sequences
                all_chains.append(chain_letter)
    chain_seqs = [protein[f"seq_chain_{letter}"] for letter in all_chains]
    chain_lengths = [len(seq) for seq in chain_seqs]
    chain_idxs = [
        [str(idx) for idx in range(1, length + 1)] for length in chain_lengths
    ]

    # Calculate fixed positions (everything not designable)
    fixed_positions: dict[str, list[int]] = {}
    for i, chain in enumerate(all_chains):
        if chain not in design_positions:
            # Entire chain is fixed
            fixed_positions[chain] = [int(x) for x in chain_idxs[i]]
        else:
            # Check if any positions are fixed
            if design_positions[chain] != chain_idxs[i]:
                fixed_pos = set(chain_idxs[i]).difference(set(design_positions[chain]))
                fixed_pos_list = sorted([int(x) for x in fixed_pos])
                if fixed_pos_list:
                    fixed_positions[chain] = fixed_pos_list

    if not fixed_positions:
        return None

    return {protein["name"]: fixed_positions}


def make_tied_positions_dict(
    config: SingleStateConfig, protein: dict[str, Any]
) -> dict[str, list[dict[str, list[int]]]] | None:
    """Create tied positions dictionary from symmetric constraints.

    Tied positions are sampled together during inference.

    Args:
        config: The SingleStateConfig with symmetric residue groups.
        protein: Protein dictionary from StructureDatasetPDB.

    Returns:
        Dictionary mapping protein name -> list of tied position dicts.
        Returns None if no symmetric constraints.
    """
    if not config.symmetric:
        return None

    dict_list: list[dict[str, list[int]]] = []
    for res_group in config.symmetric:
        tmp_dict: dict[str, list[int]] = {}
        for res_item in res_group:
            # Parse "A10" -> ("A", 10)
            split_item = re.split(r"(\d+)", res_item)
            chain_id = str(split_item[0])
            res_idx = int(split_item[1])
            tmp_dict[chain_id] = [res_idx]
        dict_list.append(tmp_dict)

    return {protein["name"]: dict_list}


def make_omit_aa_dict(
    config: SingleStateConfig, protein: dict[str, Any], experimental: bool = False
) -> dict[str, dict[str, list[list[int | str]]]]:
    """Create omit amino acids dictionary from MutTo specifications.

    Args:
        config: The SingleStateConfig with designable residues and MutTo specs.
        protein: Protein dictionary from StructureDatasetPDB.
        experimental: Whether to use experimental MutTo parsing.

    Returns:
        Dictionary mapping protein name -> chain -> list of [resid, omit_AAs] pairs.
    """
    omit_aa_dict: dict[str, list[list[int | str]]] = {}

    # Only iterate over chains that exist in the protein with non-empty sequences
    for key in protein:
        if key.startswith("seq_chain_"):
            chain_letter = key[10:]
            chain_seq = protein[key]
            if chain_seq:  # Only non-empty sequences
                omit_aa_dict[chain_letter] = []
                for res in config.designable:
                    if res.chain == chain_letter and res.MutTo != "all":
                        res_dict = {"MutTo": res.MutTo, "WTAA": res.WTAA}
                        omit_aas = parse_mutate_to(res_dict, experimental=experimental)
                        omit_aa_dict[chain_letter].append([res.resid, omit_aas])

    return {protein["name"]: omit_aa_dict}


def transform_inputs(
    config: SingleStateConfig, protein: dict[str, Any], experimental: bool = False
) -> tuple[
    None,
    dict[str, dict[str, list[int]]] | None,
    None,
    dict[str, dict[str, list[list[int | str]]]],
    None,
    dict[str, list[dict[str, list[int]]]] | None,
    None,
]:
    """Transform SingleStateConfig into model input dictionaries.

    This is the main entry point for converting design specifications
    into the format required by tied_featurize().

    Args:
        config: The SingleStateConfig with design specifications.
        protein: Protein dictionary from StructureDatasetPDB.
        experimental: Whether to use experimental MutTo parsing.

    Returns:
        A 7-tuple of:
        - chain_id_dict: Always None (handled by fixed_positions)
        - fixed_positions_dict: Positions to keep fixed
        - pssm_dict: Always None (not yet supported)
        - omit_AA_dict: Amino acids to exclude per position
        - bias_AA_dict: Always None (not yet supported)
        - tied_positions_dict: Symmetric positions to sample together
        - bias_by_res_dict: Always None (not yet supported)
    """
    chain_id_dict = None
    fixed_positions_dict = make_fixed_positions_dict(config, protein)
    pssm_dict = None
    omit_aa_dict = make_omit_aa_dict(config, protein, experimental=experimental)
    bias_aa_dict = None
    tied_positions_dict = make_tied_positions_dict(config, protein)
    bias_by_res_dict = None

    return (
        chain_id_dict,
        fixed_positions_dict,
        pssm_dict,
        omit_aa_dict,
        bias_aa_dict,
        tied_positions_dict,
        bias_by_res_dict,
    )

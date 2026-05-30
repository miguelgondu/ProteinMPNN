from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import typer

from proteinmpnn.data.single_state import SingleStateDesignInput
from proteinmpnn.data.structure.dataset import StructureDatasetPDB
from proteinmpnn.model.proteinmpnn import ProteinMPNN
from proteinmpnn.utils.constants import AA, HIDDEN_DIM, NUM_LAYERS, WEIGHTS_PATH

app = typer.Typer()


@app.command()
def run_single(
    pdb_path: Path,
    model_name: Annotated[
        Literal[
            "v_48_002",
            "v_48_010",
            "v_48_020",
            "v_48_030",  # vanilla models
            "ca_48_002",
            "ca_48_010",
            "ca_48_020",  # CA models
            "s_48_002",
            "s_48_010",
            "s_48_020",
            "s_48_030",  # soluble models
        ],
        typer.Option,
    ] = "v_48_020",
    designable_residues: str = "",
    symmetric_residues: str = "",
    cluster_center: str = "",
    cluster_radius: float = 10.0,  # TODO(miguel): Which units?
    backbone_noise: float = 0.0,
    num_seq_per_target: int = 5,
    batch_size: int = 1,
    temperature: float = 0.1,
):
    ckpt_path = WEIGHTS_PATH / f"{model_name}.pt"

    if not ckpt_path.exists():
        raise ValueError("...")

    omit_AAs = np.array([A == "X" for A in AA]).astype(np.float32)

    pdbs_dataset = StructureDatasetPDB.from_pdb_dir(pdb_path.parent)
    residue_designability = SingleStateDesignInput(
        pdb_path,
        designable_res=designable_residues,
        default_design_setting="all",
        symmetric_res=symmetric_residues,
        cluster_center=cluster_center,
        cluster_radius=cluster_radius,
    )

    ckpt = torch.load(
        ckpt_path,
        map_location=(
            device := (
                torch.device("cuda")
                if torch.cuda.is_available()
                else (
                    torch.device("mps")
                    if torch.mps.is_available()
                    else torch.device("cpu")
                )
            )
        ),
    )
    num_edges = ckpt["num_edges"]
    model = ProteinMPNN(
        num_letters=21,
        node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        augment_eps=backbone_noise,
        k_neighbors=num_edges,
    )
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(model)


if __name__ == "__main__":
    app()

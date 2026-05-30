import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from proteinmpnn.utils.logging import get_logger
from proteinmpnn.utils.pdb import parse_PDB

logger = get_logger(f"ProteinMPNN.{__file__}")


class StructureDataset:
    def __init__(
        self,
        jsonl_filepath: Path,
        *,
        verbose: bool = True,
        truncate: int | None = None,
        max_length: int = 100,
        alphabet: str | Sequence[str] = "ACDEFGHIKLMNPQRSTVWYX-",
    ):
        alphabet_set = set(alphabet)
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        with jsonl_filepath.open() as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry["seq"]
                name = entry["name"]

                # Check if in alphabet
                bad_chars = set(seq).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry["seq"]) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count["too_long"] += 1
                else:
                    logger.info(name, bad_chars, entry["seq"])
                    discard_count["bad_chars"] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    logger.info(
                        f"{len(self.data)} entries ({i + 1} loaded) in {elapsed:.1f} s"
                    )

            logger.info("Discarded during loading: ", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB:
    def __init__(
        self,
        pdb_dict_list: list[dict[str, Any]],
        *,
        verbose: bool = True,
        truncate: int | None = None,
        max_length: int = 100,
        alphabet: str | Sequence[str] = "ACDEFGHIKLMNPQRSTVWYX-",
    ):
        alphabet_set = set(alphabet)
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry["seq"]

            bad_chars = set(seq).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry["seq"]) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count["too_long"] += 1
            else:
                discard_count["bad_chars"] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                logger.info(
                    f"{len(self.data)} entries ({i + 1} loaded) in {elapsed:.1f} s"
                )

        logger.info("Discarded during loading: ", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_pdb_dir(cls, pdb_dir: str | Path) -> "StructureDatasetPDB":
        """Parses pdb directory and returns a dataset containing coordinates
        of every parsed protein chain.

        Input:
            pdb_dir (str): A string of the path where all input pdb files are
                contained.

        Output:
            StructureDatasetPDB: A dataset containing the parsed proteins and
                their backbone coordinates.
        """
        if not isinstance(pdb_dir, Path):
            pdb_dir = Path(pdb_dir)

        # if os.path.isdir(pdb_dir):
        if not pdb_dir.is_dir():
            raise ValueError(f"Could not find pdb_dir: {pdb_dir}")

        # Find all pdb files
        pdb_files = list(pdb_dir.glob("*.pdb"))
        if len(pdb_files) == 0:
            raise ValueError(f"No .pdb files detected in pdb_dir: {pdb_dir}")

        # Parse every pdb file and add parsed dict to overall list
        pdb_dict_list = []
        for pdb_file in pdb_files:
            pdb_dict_list += parse_PDB(pdb_file)

        # Construct dataset from pdb_dict_list
        return cls(pdb_dict_list, max_length=20_000)


def get_pdb_dataset(pdb_dir: str | Path) -> StructureDatasetPDB:
    """Parses pdb directory and returns a dataset containing coordinates
    of every parsed protein chain.

    Input:
        pdb_dir (str): A string of the path where all input pdb files are
            contained.

    Output:
        StructureDatasetPDB: A dataset containing the parsed proteins and
            their backbone coordinates.
    """
    if not isinstance(pdb_dir, Path):
        pdb_dir = Path(pdb_dir)

    # if os.path.isdir(pdb_dir):
    if not pdb_dir.is_dir():
        raise ValueError(f"Could not find pdb_dir: {pdb_dir}")

    # Find all pdb files
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if len(pdb_files) == 0:
        raise ValueError(f"No .pdb files detected in pdb_dir: {pdb_dir}")

    # Parse every pdb file and add parsed dict to overall list
    pdb_dict_list = []
    for pdb_file in pdb_files:
        pdb_dict_list += parse_PDB(pdb_file)

    # Construct dataset from pdb_dict_list
    return StructureDatasetPDB(pdb_dict_list, max_length=20_000)

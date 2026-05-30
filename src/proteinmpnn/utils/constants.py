from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent

CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
AA3 = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "XXX",
]
AA = list("ACDEFGHIKLMNPQRSTVWYX")

AA3_TO_AA = dict(zip(AA3, AA))

MODEL_NAMES = [
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
]

INIT_ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
EXTRA_ALPHABET = [str(item) for item in range(300)]
CHAIN_ALPHABET = INIT_ALPHABET + EXTRA_ALPHABET

WEIGHTS_PATH = ROOT_DIR / "run" / "model_weights"

HIDDEN_DIM = 128
NUM_LAYERS = 3
N_POINTS = 8

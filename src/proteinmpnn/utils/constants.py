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

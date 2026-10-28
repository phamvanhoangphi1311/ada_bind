"""Fixed dataset / task / chemistry constants shared by every module."""

TASKS = ("metal_binding", "nuclear_binding", "small_binding")
ALL_LABELS = TASKS + ("any_ligand_binding_sites",)
TASK_SHORT = {"metal": "metal_binding", "nuclear": "nuclear_binding", "small": "small_binding"}

DEV = "DevSet1010"
TEST300 = "TestSet300"
TEST46 = "TestSetNew46"
SPLITS = (DEV, TEST300, TEST46)
EXTERNAL_SPLITS = (TEST300, TEST46)
SPLIT_SIZES = {DEV: 1010, TEST300: 300, TEST46: 46}

# GEAR backend store keys.
STORE_KEY = {DEV: "train", TEST300: "test300", TEST46: "test46"}
SPLIT_OF_STORE = {v: k for k, v in STORE_KEY.items()}
# Directory aliases used by older caches.
SPLIT_ALIASES = {DEV: (DEV, "train"), TEST300: (TEST300, "test300"), TEST46: (TEST46, "test46")}

AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

CHARGE = {"K": 1.0, "R": 1.0, "H": 0.10, "D": -1.0, "E": -1.0}

# Kyte-Doolittle hydropathy.
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Feature blocks provided by the GEAR backend.
GEAR_BLOCK_DIMS = {
    "AA": 20, "ESM": 1280, "SPATIAL": 112, "LOCAL": 8, "ATOM": 15, "CHEM": 24,
    "ACCESS": 10, "SHELL2": 12, "ENV": 8, "CONF": 4,
    "METALSPEC": 18, "ELEC3D": 8, "TOPO": 12, "SOLV": 10, "SEQMOTIF": 10,
    "SEQCTX": 182, "EVO": 48, "ESMIF": 512,
}
MECH1 = ["ACCESS", "SHELL2", "ENV", "CONF"]
MULTI2 = ["METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF"]
EXPLICIT_GEAR_BLOCKS = [
    "AA", "SPATIAL", "LOCAL", "ATOM", "CHEM", "ACCESS", "SHELL2", "ENV", "CONF",
    "METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF", "SEQCTX",
]

PRIMARY_METRIC = "AUPRC"
DEFAULT_SEED = 2026
MIN_DEV_GAIN = 0.002  # complexity-aware promotion margin used everywhere

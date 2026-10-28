"""Single-protein inference with the final models: what an application calls.

    from pbind import Predictor
    P = Predictor(final_model_dir, weights_dir, profile="final")       # load once (e.g. at server start)
    result = P.predict(sequence, "metal", pdb=pdb_text)               # dict, format of docs/APP_INTEGRATION.md

Every step reproduces the training pipeline:
  * structure  the ESMFold PDB as given (GEAR blocks: ATOM/CHEM/MULTI2/ACCESS) and its exact-sequence chain-A copy
               (NA_* / SM_* blocks, patch / pocket banks), exactly like the training caches;
  * embeddings ESM-2 (fp32 forward, stored fp16), ProtT5 (fp16 on GPU / fp32 on CPU, stored fp16), ESM-IF (512-d,
               computed by the caller in its own environment, see docs/FOUNDATION_MODELS.md, or in-process when
               ``fair-esm`` is importable); ESM-IF enters the metal MLP as float32 and the hypergraphs fp16-rounded;
  * models     final_model/manifest.json: profile terms, weights, thresholds; every checkpoint is sha1-checked
               against the manifest before it is loaded.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

TASKS = ("metal", "nuclear", "small")
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN, MAX_TRAINED_LEN = 30, 1000, 813
EMB_DIMS = {"ESM": 1280, "PROTT5": 1024, "ESMIF": 512}
FOLDER = {"esm2": "esm2_t33_650M_UR50D", "prott5": "prot_t5_xl_half_uniref50-enc", "esmif": "esm_if1"}
GROUP = {"nuclear": "patch", "small": "pocket"}
TASK_BLOCKS = {  # hand-crafted blocks per task: (feature group, structure the group reads)
    "metal": (("ATOM_CHEM", "raw"), ("MULTI2", "raw")),
    "nuclear": (("MECH1", "raw"), ("NA_V1", "chain_a")),
    "small": (("MECH1", "raw"), ("SM_V1", "chain_a")),
}


class InputError(ValueError):
    """Invalid sequence / PDB / embedding (an application should answer HTTP 400)."""


def clean_sequence(text: str) -> str:
    """Upper-case single sequence; a FASTA header and whitespace are removed."""
    lines = [ln.strip() for ln in str(text).strip().splitlines()]
    seq = "".join(ln for ln in lines if ln and not ln.startswith(">")).replace(" ", "").upper()
    bad = sorted(set(seq) - AA20)
    if bad:
        raise InputError(f"sequence contains non-standard residues {bad}; only {''.join(sorted(AA20))} are allowed")
    if not MIN_LEN <= len(seq) <= MAX_LEN:
        raise InputError(f"sequence length {len(seq)} outside [{MIN_LEN}, {MAX_LEN}]")
    return seq


def _sha16(fp: Path) -> str:
    return hashlib.sha1(Path(fp).read_bytes()).hexdigest()[:16]


class Predictor:
    def __init__(self, final_model_dir, weights_dir=None, profile: str = "final", device=None):
        self.root = Path(final_model_dir)
        raw = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        if "profiles" not in self.manifest or profile not in self.manifest["profiles"]:
            raise ValueError(f"{self.root}/manifest.json has no profile {profile!r}; use the manifest shipped with pbind")
        self.profile = profile
        self.model_version = hashlib.sha1(raw).hexdigest()[:12]
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._ckpt = {}
        self._esm = self._t5 = self._esmif = None

    # ------------------------------------------------------------------ public API
    def needs(self, task: str) -> tuple[str, ...]:
        """Embeddings a task needs (metal: ESM + ESMIF; nuclear / small: + PROTT5)."""
        return ("ESM", "ESMIF") if task == "metal" else ("ESM", "ESMIF", "PROTT5")

    def predict(self, sequence: str, binding_type: str, pdb: str, embeddings: dict | None = None,
                return_pdb: bool = True) -> dict:
        """``pdb``: ESMFold PDB text (or a path). ``embeddings``: optional {"ESM", "ESMIF", "PROTT5"} (L, d) arrays
        computed by the caller; the missing ones are computed here (ESMIF only if fair-esm is importable)."""
        task = str(binding_type).strip().lower()
        if task not in TASKS:
            raise InputError(f"binding_type must be one of {TASKS}")
        seq = clean_sequence(sequence)
        from ..data.structures import _coords_cached

        try:
            with tempfile.TemporaryDirectory() as tmp:
                raw_pdb, chain_pdb, plddt, pdb_text = self._structure(seq, pdb, Path(tmp))
                emb = self.embeddings(seq, chain_pdb, self.needs(task), embeddings)
                feats, banks = self.features(task, seq, raw_pdb, chain_pdb)
                prob = self._run(task, seq, feats, banks, emb)
        finally:
            _coords_cached.cache_clear()  # memoized by (path, sequence): temp paths are reused
        terms_thr = self.manifest["profiles"][self.profile][task]["threshold"]
        call = prob >= terms_thr
        warnings = []
        if len(seq) > MAX_TRAINED_LEN:
            warnings.append(f"length {len(seq)} > {MAX_TRAINED_LEN} (longest training protein)")
        if float(np.mean(plddt)) < 0.7:
            warnings.append(f"low mean pLDDT {np.mean(plddt):.2f}: structure-based predictions are less reliable")
        return {
            "sequence": seq, "length": len(seq), "binding_type": task, "profile": self.profile,
            "model_version": self.model_version, "threshold": terms_thr,
            "probability": [float(x) for x in prob], "binding": [bool(x) for x in call],
            "binding_residues": [{"index": i + 1, "aa": seq[i], "probability": float(prob[i])}
                                 for i in np.flatnonzero(call)],
            "structure": {"plddt": [float(x) for x in plddt], **({"pdb": pdb_text} if return_pdb else {})},
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ structure
    def _structure(self, seq: str, pdb, tmp: Path):
        from ..data.structures import normalize_exact_chain_A, standard_residues, _parser

        text = str(pdb)
        if isinstance(pdb, Path) or ("\n" not in text and len(text) < 4096 and Path(text).is_file()):
            text = Path(pdb).read_text()
        if "ATOM" not in text:
            raise InputError("pdb is empty or not a PDB file")
        raw = tmp / "query.pdb"
        raw.write_text(text)
        chain = tmp / "query_chainA.pdb"
        try:
            normalize_exact_chain_A(raw, seq, chain)
        except Exception as e:
            raise InputError(f"no chain of the PDB matches the sequence exactly ({e})") from None
        rr = standard_residues(next(_parser().get_structure("q", str(chain)).get_models())["A"])
        plddt = np.array([float(r["CA"].bfactor) if "CA" in r else 0.0 for r in rr])
        return raw, chain, plddt, text

    # ------------------------------------------------------------------ embeddings
    def embeddings(self, seq: str, chain_pdb: Path, names, given: dict | None = None) -> dict:
        given = dict(given or {})
        out = {}
        for name in names:
            arr = given.get(name)
            if arr is None:
                arr = {"ESM": self.esm2, "PROTT5": self.prott5, "ESMIF": lambda s: self.esmif(chain_pdb, s)}[name](seq)
            arr = np.asarray(arr)
            if arr.shape != (len(seq), EMB_DIMS[name]):
                raise InputError(f"{name} embedding has shape {arr.shape}, expected {(len(seq), EMB_DIMS[name])}")
            if not np.isfinite(arr).all():
                raise InputError(f"{name} embedding contains NaN/inf")
            out[name] = arr
        return out

    def _weights(self, key: str) -> Path:
        if self.weights_dir is None:
            raise RuntimeError(f"weights_dir is required to compute {key} embeddings")
        return self.weights_dir / FOLDER[key]

    def esm2(self, seq: str) -> np.ndarray:
        from ..features.esm import embed_sequences, load_esm_model

        if self._esm is None:
            self._esm = load_esm_model(str(self._weights("esm2")), self.device)
        tok, model = self._esm
        return embed_sequences([("q", seq)], tok, model, self.device, batch_size=1, progress=False)["q"]

    def prott5(self, seq: str) -> np.ndarray:
        if self._t5 is None:
            from transformers import T5EncoderModel

            d = self._weights("prott5")
            try:
                from transformers import T5Tokenizer

                tok = T5Tokenizer.from_pretrained(str(d), do_lower_case=False)
                ids_fn = lambda clean: tok(" ".join(clean), add_special_tokens=True)["input_ids"]
                unk = tok.unk_token_id
            except Exception:  # transformers without a usable slow tokenizer: same SentencePiece model directly
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor(model_file=str(d / "spiece.model"))
                ids_fn = lambda clean: [sp.piece_to_id(f"▁{a}") for a in clean] + [sp.eos_id()]
                unk = sp.unk_id()
            model = T5EncoderModel.from_pretrained(str(d)).to(self.device).eval()
            model = model.half() if self.device.type == "cuda" else model.float()  # training: fp16 on GPU
            self._t5 = (ids_fn, unk, model)
        ids_fn, unk, model = self._t5
        clean = re.sub(r"[UZOB]", "X", seq)
        ids = ids_fn(clean)
        if len(ids) != len(seq) + 1 or unk in ids[:-1]:
            raise RuntimeError(f"ProtT5 tokenization gave {len(ids)} tokens for {len(seq)} residues")
        ids = torch.tensor(ids, dtype=torch.long)[None].to(self.device)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        return h[0, :len(seq)].float().cpu().numpy().astype(np.float16)

    def esmif(self, chain_pdb: Path, seq: str) -> np.ndarray:
        """In-process ESM-IF (needs the fair-esm environment of docs/FOUNDATION_MODELS.md section 5)."""
        try:
            import esm
            import esm.inverse_folding
            from esm.inverse_folding.util import CoordBatchConverter
        except Exception as e:
            raise RuntimeError("ESM-IF needs fair-esm (separate environment): compute the (L, 512) embedding there "
                               "(scripts/esmif_embed.py) and pass embeddings={'ESMIF': ...}") from e
        if self._esmif is None:
            fp = self._weights("esmif") / "esm_if1_gvp4_t16_142M_UR50.pt"
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(fp))
            self._esmif = (model.eval().to(self.device), alphabet, CoordBatchConverter)
        model, alphabet, conv = self._esmif
        structure = esm.inverse_folding.util.load_structure(str(chain_pdb), "A")
        coords, pdb_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        if pdb_seq != seq:
            raise InputError("ESM-IF: PDB chain A does not match the sequence")
        with torch.no_grad():
            c, conf, _, _, pad = conv(alphabet)([(coords, None, None)], device=self.device)
            enc = model.encoder.forward(c, pad, conf, return_all_hiddens=False)
            return enc["encoder_out"][0][1:-1, 0].cpu().numpy()

    # ------------------------------------------------------------------ hand-crafted features + groups
    def features(self, task: str, seq: str, raw_pdb: Path, chain_pdb: Path):
        from ..data.structures import structure_coords
        from ..features import FEATURE_GROUPS, Protein
        from ..groups.bank import GROUP_BANKS

        feats = {}
        for group, source in TASK_BLOCKS[task]:
            p = Protein("query", seq, "query", raw_pdb if source == "raw" else chain_pdb)
            for block, arr in FEATURE_GROUPS[group].fn(p).items():
                arr = np.asarray(arr, dtype=np.float32)
                if arr.shape[0] != len(seq):
                    raise InputError(f"{group}/{block}: {arr.shape[0]} residues for a {len(seq)}-residue sequence")
                feats[block] = arr
        banks = {}
        if task in GROUP:
            coords = structure_coords(chain_pdb, seq)
            for kind in ("v1", "ms"):
                spec = GROUP_BANKS[f"{task}_{GROUP[task]}_{kind}"]
                er, eg, gf = spec.build(coords, seq, **spec.params)
                banks[kind] = (np.asarray(er, np.int64), np.asarray(eg, np.int64), np.asarray(gf, np.float32), spec)
        return feats, banks

    # ------------------------------------------------------------------ models
    def _files(self, member: str):
        info = self.manifest["members"][member]
        for f, want in sorted(info["sha1_16"].items()):
            if f.endswith(".pt"):
                fp = self.root / member / f
                if _sha16(fp) != want:
                    raise RuntimeError(f"{fp}: checksum differs from manifest.json (corrupt or wrong file)")
                yield fp

    def _metal_member(self, member: str):
        from ..models.residue import ResidueMLP

        if member not in self._ckpt:
            folds = []
            for fp in self._files(member):  # sha1-verified, so the pickle (numpy scaler) is the shipped one
                ck = torch.load(fp, map_location="cpu", weights_only=False)
                d = int(np.asarray(ck["scaler"]["mean"]).shape[0])
                m = ResidueMLP(d, 0.2)
                m.load_state_dict(ck["state_dict"])
                folds.append((m.to(self.device).eval(), list(ck["blocks"]),
                              np.asarray(ck["scaler"]["mean"], np.float32), np.asarray(ck["scaler"]["std"], np.float32)))
            self._ckpt[member] = folds
        return self._ckpt[member]

    def _hg_member(self, task: str, member: str, dims: dict):
        from ..models.hypergraph import GroupHypergraph

        if member not in self._ckpt:
            kind = member.split("/")[1].split("_")[0]  # v1 | ms
            spec_dim = {"nuclear": {"v1": 18, "ms": 24}, "small": {"v1": 22, "ms": 23}}[task][kind]
            folds = []
            for fp in self._files(member):
                sd = torch.load(fp, map_location="cpu", weights_only=True)
                m = GroupHypergraph(dims, group=GROUP[task], static_dim=spec_dim, mode="attention", latent=192)
                m.load_state_dict(sd)  # strict: modality set / dimensions must match the checkpoint
                folds.append(m.to(self.device).eval())
            self._ckpt[member] = (kind, folds)
        return self._ckpt[member]

    @torch.no_grad()
    def _member_prob(self, task: str, member: str, feats: dict, banks: dict, emb: dict) -> np.ndarray:
        if task == "metal":
            blocks = {**feats, "ESM": emb["ESM"].astype(np.float16).astype(np.float32),
                      "ESMIF": emb["ESMIF"].astype(np.float32)}
            ps = []
            for m, order, mean, std in self._metal_member(member):
                X = np.concatenate([np.asarray(blocks[b], np.float32) for b in order], axis=1)
                X = ((X - mean) / std).astype(np.float32)
                ps.append(torch.sigmoid(m(torch.from_numpy(X).to(self.device)).float()).cpu().numpy().astype(np.float64))
            return np.mean(np.stack(ps), axis=0)
        blocks = {**feats, "ESM": emb["ESM"].astype(np.float16).astype(np.float32)}
        if "PROTT5" in emb:
            blocks["PROTT5"] = emb["PROTT5"].astype(np.float16).astype(np.float32)
        if "ESMIF" in emb:
            blocks["ESMIF_EF"] = emb["ESMIF"].astype(np.float16).astype(np.float32)  # hypergraphs: fp16-rounded
        spec = self.manifest["members"][member]["modalities"]
        mods = {k: np.concatenate([blocks[b] for b in bl], axis=1).astype(np.float32) for k, bl in spec.items()}
        dims = {k: v.shape[1] for k, v in mods.items()}
        kind, folds = self._hg_member(task, member, dims)
        er, eg, gf, _ = banks[kind]
        args = ({k: torch.from_numpy(v).to(self.device) for k, v in mods.items()}, torch.from_numpy(er).to(self.device),
                torch.from_numpy(eg).to(self.device), torch.from_numpy(gf).to(self.device))
        ps = [torch.sigmoid(m(*args)).float().cpu().numpy().astype(np.float64) for m in folds]
        return np.mean(np.stack(ps), axis=0)

    def _run(self, task: str, seq: str, feats: dict, banks: dict, emb: dict) -> np.ndarray:
        prob = 0
        for term in self.manifest["profiles"][self.profile][task]["terms"]:
            if term["weight"]:
                parts = [self._member_prob(task, m, feats, banks, emb) for m in term["members"]]
                prob = prob + term["weight"] * np.mean(parts, axis=0)
        return np.asarray(prob, dtype=np.float64)


def predict(sequence: str, binding_type: str, pdb: str, final_model_dir, weights_dir=None, profile: str = "final",
            embeddings: dict | None = None, device=None) -> dict:
    """One-call convenience (loads the models every call; servers should keep one ``Predictor``)."""
    return Predictor(final_model_dir, weights_dir, profile, device).predict(sequence, binding_type, pdb, embeddings)

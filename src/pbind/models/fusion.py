"""Per-residue multimodal fusion models (input: dict modality -> (N, d) tensor).

State-dict key names are identical to the original notebooks, so fold checkpoints
saved under results/*/custom_states/<arch>/fold*.pt load directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _enc(d: int, latent: int, drop: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop))


def _head(latent: int, hidden: int, drop: float, norm: bool = True) -> nn.Sequential:
    layers = [nn.LayerNorm(latent)] if norm else []
    return nn.Sequential(*layers, nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))


class ConcatMLP(nn.Module):
    def __init__(self, dims, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.norm = nn.ModuleDict({k: nn.LayerNorm(d) for k, d in dims.items()})
        self.head = nn.Sequential(
            nn.Linear(sum(dims.values()), 512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(drop),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(torch.cat([self.norm[k](x[k]) for k in self.keys], dim=-1)).squeeze(-1)


class ProjectedConcat(nn.Module):
    def __init__(self, dims, latent=192, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: _enc(d, latent, drop) for k, d in dims.items()})
        self.head = nn.Sequential(nn.Linear(latent * len(dims), 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(torch.cat([self.enc[k](x[k]) for k in self.keys], dim=-1)).squeeze(-1)


class GatedFusion(nn.Module):
    def __init__(self, dims, latent=192, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: _enc(d, latent, drop) for k, d in dims.items()})
        self.gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
        self.head = _head(latent, 192, drop)

    def encode(self, x):
        zz = [self.enc[k](x[k]) for k in self.keys]
        Z = torch.stack(zz, dim=1)
        G = torch.softmax(torch.cat([self.gate[k](z) for k, z in zip(self.keys, zz)], dim=1), dim=1).unsqueeze(-1)
        return (Z * G).sum(dim=1), G.squeeze(-1)

    def forward(self, x):
        h, _ = self.encode(x)
        return self.head(h).squeeze(-1)


class ModalityAttention(nn.Module):
    def __init__(self, dims, latent=192, heads=6, drop=0.2):
        super().__init__()
        assert latent % heads == 0
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: _enc(d, latent, drop) for k, d in dims.items()})
        self.attn = nn.MultiheadAttention(latent, heads, dropout=drop, batch_first=True)
        self.head = _head(latent, 192, drop)

    def forward(self, x):
        Z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
        A, _ = self.attn(Z, Z, Z, need_weights=False)
        return self.head((Z + A).mean(dim=1)).squeeze(-1)


class ResidualFusion(nn.Module):
    """Gated fusion of the base modalities + small learned residual from one structural modality.

    nuclear: residual_key="PATCH",  prefix="patch"   (PatchResidualFusion)
    small:   residual_key="POCKET", prefix="pocket"  (PocketResidualFusion)
    """

    def __init__(self, dims, residual_key="PATCH", prefix="patch", latent=192, drop=0.2):
        super().__init__()
        if residual_key not in dims:
            raise ValueError(f"ResidualFusion requires {residual_key} modality")
        self.residual_key, self._prefix = residual_key, prefix
        self.base_keys = sorted(k for k in dims if k != residual_key)
        self.base_enc = nn.ModuleDict({k: _enc(d, latent, drop) for k, d in dims.items() if k != residual_key})
        setattr(self, f"{prefix}_enc", nn.Sequential(
            nn.LayerNorm(dims[residual_key]), nn.Linear(dims[residual_key], latent), nn.GELU(), nn.Dropout(drop),
            nn.Linear(latent, latent)))
        self.base_gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in self.base_keys})
        self.alpha_logit = nn.Parameter(torch.tensor(-2.944, dtype=torch.float32))  # sigmoid ≈ 0.05
        self.head = _head(latent, 192, drop)

    def forward(self, x):
        zz = [self.base_enc[k](x[k]) for k in self.base_keys]
        Z = torch.stack(zz, dim=1)
        G = torch.softmax(torch.cat([self.base_gate[k](z) for k, z in zip(self.base_keys, zz)], dim=1), dim=1)
        h = (Z * G.unsqueeze(-1)).sum(dim=1)
        res = getattr(self, f"{self._prefix}_enc")(x[self.residual_key])
        return self.head(h + torch.sigmoid(self.alpha_logit) * res).squeeze(-1)


class LatentExpertMoE(nn.Module):
    """Gated modality fusion -> shared trunk -> soft mixture of experts.

    (LatentNAExpertMoE for nuclear, LatentLigandMoE for small — identical weights layout.)
    """

    def __init__(self, dims, latent=192, experts=4, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.enc = nn.ModuleDict({k: _enc(d, latent, drop) for k, d in dims.items()})
        self.mod_gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
        self.shared = nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU())
        self.experts = nn.ModuleList([_head(latent, 128, drop, norm=False) for _ in range(experts)])
        self.expert_gate = nn.Linear(latent, experts)

    def forward(self, x):
        zz = [self.enc[k](x[k]) for k in self.keys]
        Z = torch.stack(zz, dim=1)
        G = torch.softmax(torch.cat([self.mod_gate[k](z) for k, z in zip(self.keys, zz)], dim=1), dim=1)
        h = self.shared((Z * G.unsqueeze(-1)).sum(dim=1))
        logits = torch.cat([e(h) for e in self.experts], dim=1)
        return (logits * torch.softmax(self.expert_gate(h), dim=1)).sum(dim=1)


def fusion_factory(dims, residual_key: str, prefix: str, residual_name: str, moe_name: str) -> dict:
    """Architecture registry in the order the notebooks screened them."""
    reg = {
        "ConcatMLP": lambda d: ConcatMLP(d),
        "ProjectedConcat": lambda d: ProjectedConcat(d),
        "GatedFusion": lambda d: GatedFusion(d),
        "ModalityAttention": lambda d: ModalityAttention(d),
        moe_name: lambda d: LatentExpertMoE(d),
    }
    if residual_key in dims:
        reg[residual_name] = lambda d: ResidualFusion(d, residual_key=residual_key, prefix=prefix)
    return reg

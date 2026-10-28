"""Residue <-> group (surface patch / pocket) bipartite hypergraph models.

GroupHypergraph   = V1  NuclearPatchHypergraph / SmallPocketHypergraph
                    (and the *_MS variants: same model, different static_dim)
GroupMessageV2    = V2  NuclearPatchV2 (none | cross) / SmallPocketV2 (none | mean | cross)

``group`` ("patch" | "pocket") only decides sub-module names, keeping the
state_dict keys of the original notebooks (patch_static, pocket_mlp, ...).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def index_mean(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_zeros((n, x.shape[-1]))
    cnt = x.new_zeros((n, 1))
    if index.numel():
        out = out.index_add(0, index, x)
        cnt = cnt.index_add(0, index, x.new_ones((len(index), 1)))
    return out / cnt.clamp_min(1.0)


def index_max(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return x.new_zeros((0, x.shape[-1]))
    base = x.new_full((n, x.shape[-1]), -1e9)
    if index.numel():
        # out-of-place scatter_reduce: autograd safe
        out = base.scatter_reduce(0, index[:, None].expand(-1, x.shape[-1]), x, reduce="amax", include_self=True)
    else:
        out = base
    return torch.where(out < -1e8, torch.zeros_like(out), out)


def incident_mask(edge_res: torch.Tensor, nres: int, like: torch.Tensor) -> torch.Tensor:
    return like.new_zeros((nres, 1)).index_add(0, edge_res, like.new_ones((len(edge_res), 1)))


class ModalityEncoder(nn.Module):
    """Per-modality LayerNorm/Linear/GELU encoders fused by softmax gates or self-attention."""

    def __init__(self, dims, mode="gated", latent=192, heads=6, drop=0.2):
        super().__init__()
        self.keys = sorted(dims)
        self.mode = mode
        self.enc = nn.ModuleDict({
            k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent), nn.GELU(), nn.Dropout(drop))
            for k, d in dims.items()
        })
        if mode == "gated":
            self.gate = nn.ModuleDict({k: nn.Linear(latent, 1) for k in dims})
        elif mode == "attention":
            assert latent % heads == 0
            self.attn = nn.MultiheadAttention(latent, heads, dropout=drop, batch_first=True)
        else:
            raise ValueError(mode)

    def forward(self, x):
        Z = torch.stack([self.enc[k](x[k]) for k in self.keys], dim=1)
        if self.mode == "gated":
            G = torch.cat([self.gate[k](Z[:, j, :]) for j, k in enumerate(self.keys)], dim=1)
            return (Z * torch.softmax(G, dim=1).unsqueeze(-1)).sum(dim=1)
        A, _ = self.attn(Z, Z, Z, need_weights=False)
        return (Z + A).mean(dim=1)


def _head(latent, drop):
    return nn.Sequential(nn.LayerNorm(latent), nn.Linear(latent, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 1))


class GroupHypergraph(nn.Module):
    """V1: residues -> group (mean+max+static) -> mean message back, added with a small learned alpha."""

    def __init__(self, dims, group="patch", static_dim=18, mode="gated", latent=192, moe=False,
                 experts=4, drop=0.2):
        super().__init__()
        self._g = group
        self.encoder = ModalityEncoder(dims, mode=mode, latent=latent, drop=drop)
        setattr(self, f"{group}_static", nn.Sequential(nn.LayerNorm(static_dim), nn.Linear(static_dim, 64), nn.GELU()))
        setattr(self, f"{group}_mlp", nn.Sequential(
            nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU()))
        self.message_mlp = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
        self.alpha_logit = nn.Parameter(torch.tensor(-2.944, dtype=torch.float32))  # sigmoid ≈ 0.05
        self.moe = moe
        if not moe:
            self.head = _head(latent, drop)
        else:
            self.experts = nn.ModuleList([_head(latent, drop) for _ in range(experts)])
            self.expert_gate = nn.Linear(latent, experts)

    @property
    def alpha(self) -> float:
        return float(torch.sigmoid(self.alpha_logit).detach().cpu())

    def forward(self, mods, edge_res, edge_group, group_feat):
        h = self.encoder(mods)
        nres, ngroup = h.shape[0], group_feat.shape[0]
        if ngroup and edge_res.numel():
            mean = index_mean(h[edge_res], edge_group, ngroup)
            mx = index_max(h[edge_res], edge_group, ngroup)
            stat = getattr(self, f"{self._g}_static")(group_feat)
            hg = getattr(self, f"{self._g}_mlp")(torch.cat([mean, mx, stat], dim=-1))
            msg = index_mean(hg[edge_group], edge_res, nres)
            delta = self.message_mlp(msg) * (incident_mask(edge_res, nres, h) > 0).float()
            h = h + torch.sigmoid(self.alpha_logit) * delta
        if not self.moe:
            return self.head(h).squeeze(-1)
        logits = torch.cat([e(h) for e in self.experts], dim=1)
        return (logits * torch.softmax(self.expert_gate(h), dim=1)).sum(dim=1)


class GroupMessageV2(nn.Module):
    """V2: attention residue encoder + group message (none | mean | cross-attention) with a
    residue-specific sigmoid gate (initialised ≈ 0.18)."""

    def __init__(self, dims, group="patch", static_dim=24, message_mode="cross", latent=192, heads=6, drop=0.20):
        super().__init__()
        assert message_mode in {"none", "mean", "cross"}
        self._g = group
        self.message_mode = message_mode
        self.latent = latent
        self.residue_encoder = ModalityEncoder(dims, mode="attention", latent=latent, heads=heads, drop=drop)
        setattr(self, f"{group}_static", nn.Sequential(nn.LayerNorm(static_dim), nn.Linear(static_dim, 64), nn.GELU()))
        setattr(self, f"{group}_encoder", nn.Sequential(
            nn.Linear(latent * 2 + 64, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent), nn.GELU()))
        self.q = nn.Linear(latent, latent, bias=False)
        self.k = nn.Linear(latent, latent, bias=False)
        self.v = nn.Linear(latent, latent, bias=False)
        self.delta = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, latent), nn.GELU(), nn.Dropout(drop), nn.Linear(latent, latent))
        self.gate = nn.Sequential(nn.LayerNorm(latent * 2), nn.Linear(latent * 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)
        self.head = _head(latent, drop)

    def forward(self, mods, edge_res, edge_group, group_feat, return_aux=False):
        h = self.residue_encoder(mods)
        gate_mean = h.new_tensor(0.0)
        if self.message_mode != "none" and group_feat.shape[0] > 0 and edge_res.numel() > 0:
            nres, ngroup = h.shape[0], group_feat.shape[0]
            mean = index_mean(h[edge_res], edge_group, ngroup)
            mx = index_max(h[edge_res], edge_group, ngroup)
            stat = getattr(self, f"{self._g}_static")(group_feat)
            hg = getattr(self, f"{self._g}_encoder")(torch.cat([mean, mx, stat], dim=-1))
            if self.message_mode == "mean":
                msg = index_mean(hg[edge_group], edge_res, nres)
            else:  # stable per-residue softmax over incident groups
                score = (self.q(h[edge_res]) * self.k(hg[edge_group])).sum(dim=-1) / math.sqrt(self.latent)
                max_r = score.new_full((nres,), -1e9).scatter_reduce(0, edge_res, score, reduce="amax",
                                                                     include_self=True)
                max_r = torch.where(max_r < -1e8, torch.zeros_like(max_r), max_r)
                w = torch.exp(score - max_r[edge_res])
                denom = w.new_zeros((nres, 1)).index_add(0, edge_res, w[:, None])
                msg = h.new_zeros((nres, self.latent)).index_add(0, edge_res, self.v(hg[edge_group]) * w[:, None])
                msg = msg / denom.clamp_min(1e-8)
            incident = incident_mask(edge_res, nres, h)
            g = torch.sigmoid(self.gate(torch.cat([h, msg], dim=-1))) * (incident > 0).float()
            h = h + g * self.delta(msg)
            active = incident.squeeze(-1) > 0
            if active.any():
                gate_mean = g[active].mean()
        logits = self.head(h).squeeze(-1)
        return (logits, {"gate_mean": gate_mean}) if return_aux else logits

"""Time-conditioned 3D U-Net (DDPM style) for 64^3 ROIs.

Input channels = [x_t, condition]; output = predicted noise epsilon.
Default (base 32, mults 1-2-4-4, 2 res-blocks, attention at 8^3) has ~11M
parameters and fits batch 4 at 64^3 in fp16/bf16 on a 16 GB GPU.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(ch: int, groups: int) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temb: int, dropout: float, groups: int):
        super().__init__()
        self.n1 = _norm(cin, groups)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.t = nn.Linear(temb, 2 * cout)
        self.n2 = _norm(cout, groups)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(temb))[:, :, None, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int, groups: int):
        super().__init__()
        self.n = _norm(ch, groups)
        self.heads = heads
        self.qkv = nn.Conv3d(ch, 3 * ch, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, D * H * W).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        return x + self.proj(out)


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, base_ch: int = 32, ch_mults: Sequence[int] = (1, 2, 4, 4),
                 num_res: int = 2, attn_levels: Sequence[int] = (3,), dropout: float = 0.1, groups: int = 8,
                 heads: int = 4):
        super().__init__()
        temb = base_ch * 4
        self.base_ch = base_ch
        self.temb = nn.Sequential(nn.Linear(base_ch, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans = [base_ch]
        ch = base_ch
        for lvl, m in enumerate(ch_mults):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(ch, base_ch * m, temb, dropout, groups)])
                ch = base_ch * m
                if lvl in attn_levels:
                    blk.append(AttnBlock(ch, heads, groups))
                self.down.append(blk)
                chans.append(ch)
            if lvl != len(ch_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv3d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
        self.mid1 = ResBlock(ch, ch, temb, dropout, groups)
        self.mid_attn = AttnBlock(ch, heads, groups)
        self.mid2 = ResBlock(ch, ch, temb, dropout, groups)
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(ch_mults))):
            for i in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), base_ch * m, temb, dropout, groups)])
                ch = base_ch * m
                if lvl in attn_levels:
                    blk.append(AttnBlock(ch, heads, groups))
                if lvl != 0 and i == num_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv3d(ch, ch, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(_norm(ch, groups), nn.SiLU(), nn.Conv3d(ch, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb(timestep_embedding(t, self.base_ch))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, temb)
                for extra in blk[1:]:
                    h = extra(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for blk in self.up:
            h = blk[0](torch.cat([h, hs.pop()], dim=1), temb)
            for extra in blk[1:]:
                h = extra(h)
        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

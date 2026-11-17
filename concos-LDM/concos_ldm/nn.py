"""Shared building blocks for the VAE and the latent UNet."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c, eps=1e-6)


def zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = norm(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = zero(nn.Conv2d(c, c, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class ResBlock(nn.Module):
    """Residual block with optional (timestep/class) embedding via scale-shift."""

    def __init__(self, cin, cout, emb_dim=None, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout) if emb_dim else None
        self.n2, self.drop = norm(cout), nn.Dropout(dropout)
        self.c2 = zero(nn.Conv2d(cout, cout, 3, padding=1)) if emb_dim else nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.emb is not None:
            scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

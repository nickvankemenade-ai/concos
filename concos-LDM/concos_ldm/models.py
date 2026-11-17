"""KL-VAE (first stage) and latent UNet (second stage)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import Attention, Down, ResBlock, Up, norm, timestep_embedding


class VAE(nn.Module):
    """Convolutional KL-VAE, 32x32x3 -> z_ch x 8 x 8 (downsampling factor 4)."""

    def __init__(self, ch=(64, 128, 256), z_ch=4, n_res=2):
        super().__init__()
        self.z_ch = z_ch
        enc, c = [nn.Conv2d(3, ch[0], 3, padding=1)], ch[0]
        for i, co in enumerate(ch):
            for _ in range(n_res):
                enc.append(ResBlock(c, co))
                c = co
            if i < len(ch) - 1:
                enc.append(Down(c))
        enc += [ResBlock(c, c), Attention(c), ResBlock(c, c), norm(c), nn.SiLU(), nn.Conv2d(c, 2 * z_ch, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv2d(z_ch, c, 3, padding=1), ResBlock(c, c), Attention(c), ResBlock(c, c)]
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(n_res + 1):
                dec.append(ResBlock(c, co))
                c = co
            if i > 0:
                dec.append(Up(c))
        dec += [norm(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        mu, logvar = self.encoder(x).float().chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


class _Layer(nn.Module):
    def __init__(self, block, attn):
        super().__init__()
        self.block, self.attn = block, attn

    def forward(self, x, emb):
        x = self.block(x, emb)
        return self.attn(x) if self.attn is not None else x


class UNet(nn.Module):
    """Small UNet on 8x8 latents. Class label `num_classes` is the null (unconditional) token."""

    def __init__(self, in_ch=4, ch=128, mults=(1, 2), n_res=2, attn_res=(4,), in_res=8,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        emb = ch * 4
        self.ch = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.class_emb = nn.Embedding(num_classes + 1, emb)
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down, chans, c, res = nn.ModuleList(), [ch], ch, in_res
        for i, m in enumerate(mults):
            for _ in range(n_res):
                self.down.append(_Layer(ResBlock(c, ch * m, emb, dropout), Attention(ch * m) if res in attn_res else None))
                c = ch * m
                chans.append(c)
            if i < len(mults) - 1:
                self.down.append(Down(c))
                chans.append(c)
                res //= 2
        self.mid = nn.ModuleList([_Layer(ResBlock(c, c, emb, dropout), Attention(c)),
                                  _Layer(ResBlock(c, c, emb, dropout), None)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(n_res + 1):
                self.up.append(_Layer(ResBlock(c + chans.pop(), ch * m, emb, dropout),
                                      Attention(ch * m) if res in attn_res else None))
                c = ch * m
            if i > 0:
                self.up.append(Up(c))
                res *= 2
        self.out = nn.Sequential(norm(c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y):
        """x: (B, C, 8, 8); t: (B,) in [0, 1]; y: (B,) class ids, `num_classes` = null."""
        e = self.t_mlp(timestep_embedding(t * 1000.0, self.ch)) + self.class_emb(y)
        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, e) if isinstance(layer, _Layer) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e)
        for layer in self.up:
            h = layer(torch.cat([h, hs.pop()], 1), e) if isinstance(layer, _Layer) else layer(h)
        return self.out(h)

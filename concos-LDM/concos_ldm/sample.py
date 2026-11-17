"""Generate images from a trained LDM (the hook for visualisation tools)."""

import os

import torch

from .common import NUM_CLASSES, autocast, get_device, to_unit
from .diffusion import ddim_sample
from .train import load_latents, load_ldm, load_vae


@torch.no_grad()
def decode(vae, z_raw, batch_size=500):
    dev = next(vae.parameters()).device
    out = []
    for i in range(0, len(z_raw), batch_size):
        with autocast(dev):
            out.append(to_unit(vae.decode(z_raw[i:i + batch_size].to(dev)).float()).cpu())
    return torch.cat(out)


class Sampler:
    """Bundle of VAE + LDM + latent normaliser.

        s = Sampler(work, "concos", cond=True)
        imgs, z = s.sample(labels=torch.arange(10).repeat(8), guidance=3.0)   # imgs in [0, 1]
    """

    def __init__(self, work, vae_name, cond, device=None):
        work = os.path.expanduser(work)
        self.dev = device or get_device()
        self.cond = cond
        self.vae = load_vae(work, vae_name, self.dev)
        self.model = load_ldm(work, vae_name, cond, self.dev)
        lat = load_latents(work, vae_name, self.dev)
        self.mean, self.std = lat["mean"], lat["std"]
        self.z_ch = lat["mu"].shape[1]

    @torch.no_grad()
    def sample(self, n=None, labels=None, guidance=1.0, steps=100, seed=0, chunk=1000):
        """Returns (images in [0, 1] on CPU, raw latents). Unconditional if `labels` is None."""
        if labels is None:
            labels = torch.full((n,), NUM_CLASSES, dtype=torch.long)
        labels = labels.to(self.dev)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        zs = []
        for i in range(0, len(labels), chunk):
            y = labels[i:i + chunk]
            z = ddim_sample(self.model, (len(y), self.z_ch, 8, 8), y, guidance=guidance, steps=steps,
                            device=self.dev, generator=g, autocast=autocast(self.dev))
            zs.append(z * self.std + self.mean)
        z = torch.cat(zs)
        return decode(self.vae, z), z

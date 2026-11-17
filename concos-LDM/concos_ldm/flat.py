"""v2 model family: flat-vector VAE + MLP latent diffusion.

Differences from the v1 (spatial 4x8x8) family:
  * the latent is a flat vector (default 64-d), as in the ConCoS paper and a dense-bottleneck AE;
  * the decoder has no normalisation layers, so the latent's magnitude actually reaches the image
    (v1's GroupNorm decoder divided most of it out);
  * reconstruction is MSE only (fast); both VAEs get a small Euclidean contrastive term, the ConCoS
    VAE also gets the cosine (ConCoS) term;
  * the LDM is an MLP denoiser with FiLM time/class conditioning.
Checkpoints are named vae_flat_<name>.pt, latents_flat_<name>.pt, ldm_flat_<name>_{cond,uncond}.pt,
so the v1 files are never touched.
"""

import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (NUM_CLASSES, autocast, batches, get_device, load_cifar10, random_flip, seed_all,
                     to_float, to_unit)
from .diffusion import ddim_sample, v_loss
from .loss import concos_loss, euclid_contrast
from .nn import timestep_embedding


def _log(msg, log_file=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def path(work, kind, name, cond=None):
    tag = f"_{'cond' if cond else 'uncond'}" if cond is not None else ""
    return os.path.join(work, "ckpt", f"{kind}_flat_{name}{tag}.pt")


class FlatVAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.z_dim = z_dim
        blk = lambda ci, co, k, s, p: [nn.Conv2d(ci, co, k, s, p), nn.BatchNorm2d(co), nn.SiLU()]
        self.encoder = nn.Sequential(*blk(3, 64, 3, 1, 1), *blk(64, 128, 4, 2, 1), *blk(128, 256, 4, 2, 1),
                                     *blk(256, 256, 4, 2, 1), nn.Flatten(), nn.Linear(256 * 16, 512), nn.SiLU(),
                                     nn.Linear(512, 2 * z_dim))
        # decoder without normalisation: scaling z changes the activations all the way through
        self.decoder = nn.Sequential(nn.Linear(z_dim, 512), nn.SiLU(), nn.Linear(512, 256 * 16), nn.SiLU(),
                                     nn.Unflatten(1, (256, 4, 4)),
                                     nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
                                     nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
                                     nn.Conv2d(64, 3, 3, 1, 1))

    def encode(self, x):
        mu, logvar = self.encoder(x).float().chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, z):
        return self.decoder(z)


class MLPDenoiser(nn.Module):
    """Residual MLP with FiLM conditioning on (t, class). Class `num_classes` is the null token."""

    def __init__(self, dim=64, hidden=512, depth=6, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.num_classes, self.hidden = num_classes, hidden
        self.inp = nn.Linear(dim, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.class_emb = nn.Embedding(num_classes + 1, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            b = nn.ModuleDict({"norm": nn.LayerNorm(hidden, elementwise_affine=False),
                               "film": nn.Linear(hidden, 2 * hidden),
                               "fc1": nn.Linear(hidden, 2 * hidden), "fc2": nn.Linear(2 * hidden, hidden),
                               "drop": nn.Dropout(dropout)})
            nn.init.zeros_(b["fc2"].weight)
            nn.init.zeros_(b["fc2"].bias)
            self.blocks.append(b)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, y):
        e = F.silu(self.t_mlp(timestep_embedding(t * 1000.0, 128)) + self.class_emb(y))
        h = self.inp(x)
        for b in self.blocks:
            scale, shift = b["film"](e).chunk(2, dim=1)
            u = b["norm"](h) * (1 + scale) + shift
            h = h + b["fc2"](b["drop"](F.silu(b["fc1"](u))))
        return self.out(h)


def train_vae(work, name, concos_weight, euclid_weight=0.05, margin=2.0, z_dim=64, epochs=20, batch_size=128,
              lr=1e-3, kl_weight=1e-4, seed=0, log_file=None):
    dev = get_device()
    seed_all(seed)
    (x_tr, y_tr), (x_te, _) = load_cifar10(work, dev)
    vae = FlatVAE(z_dim).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    total = epochs * (len(x_tr) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    _log(f"flat VAE[{name}] {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M params, z={z_dim}, "
         f"lambda_concos={concos_weight} w_euclid={euclid_weight} (margin {margin}) beta={kl_weight}", log_file)
    for ep in range(epochs):
        vae.train()
        sums, n, t0 = {}, 0, time.time()
        for idx in batches(len(x_tr), batch_size, dev):
            x, y = random_flip(to_float(x_tr[idx])), y_tr[idx]
            with autocast(dev):
                mu, logvar = vae.encode(x)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                rec = F.mse_loss(vae.decode(z).float(), x)
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(1).mean()
            eu = euclid_contrast(mu, y, margin)
            cc = concos_loss(mu, y)
            loss = rec + kl_weight * kl + euclid_weight * eu + concos_weight * cc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("mse", rec), ("kl", kl), ("euclid", eu), ("concos", cc)):
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        _log(f"flat VAE[{name}] ep {ep + 1}/{epochs} " + " ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
             + f" | test PSNR {psnr(vae, x_te[:2000]):.2f} ({time.time() - t0:.0f}s)", log_file)
    os.makedirs(os.path.dirname(path(work, "vae", name)), exist_ok=True)
    torch.save({"state": vae.state_dict(), "z_dim": z_dim, "concos_weight": concos_weight,
                "euclid_weight": euclid_weight, "margin": margin, "kl_weight": kl_weight, "epochs": epochs},
               path(work, "vae", name))
    return vae


@torch.no_grad()
def psnr(vae, x_uint8, bs=1000):
    vae.eval()
    out = []
    for i in range(0, len(x_uint8), bs):
        x = to_float(x_uint8[i:i + bs])
        r = vae.decode(vae.encode(x)[0]).float().clamp(-1, 1)
        out.append(-10 * torch.log10(((r - x) / 2).pow(2).flatten(1).mean(1)))
    return float(torch.cat(out).mean())


def load_vae(work, name, dev):
    ck = torch.load(path(work, "vae", name), map_location=dev)
    vae = FlatVAE(ck["z_dim"]).to(dev)
    vae.load_state_dict(ck["state"])
    return vae.eval().requires_grad_(False)


@torch.no_grad()
def encode_latents(work, name, log_file=None):
    dev = get_device()
    (x_tr, y_tr), (x_te, y_te) = load_cifar10(work, dev)
    vae = load_vae(work, name, dev)

    def enc(x, flip=False):
        out = [vae.encode(to_float(x[i:i + 1000]).flip(3) if flip else to_float(x[i:i + 1000])) for i in
               range(0, len(x), 1000)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    (ma, la), (mb, lb), (mt, lt) = enc(x_tr), enc(x_tr, True), enc(x_te)
    mu, lv = torch.cat([ma, mb]), torch.cat([la, lb])
    z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    mean = z.mean(0)
    std = (z - mean).std()
    _log(f"flat latents[{name}] dim {mu.shape[1]} |mu| mean {mu.norm(dim=1).mean():.2f} std {std:.3f} "
         f"posterior std {torch.exp(0.5 * lv).mean():.4f}", log_file)
    torch.save({"mu": mu.cpu(), "logvar": lv.cpu(), "y": torch.cat([y_tr, y_tr]).cpu(), "mu_test": mt.cpu(),
                "logvar_test": lt.cpu(), "y_test": y_te.cpu(), "mean": mean.cpu(), "std": std.cpu()},
               path(work, "latents", name))


def load_latents(work, name, dev):
    lat = torch.load(path(work, "latents", name), map_location=dev)
    lat["mean"] = lat["mean"][None]
    return lat


def train_ldm(work, name, cond, steps=30_000, batch_size=512, lr=5e-4, p_uncond=0.1, ema_decay=0.9995,
              warmup=500, seed=0, log_every=2000, log_file=None):
    dev = get_device()
    seed_all(seed)
    lat = load_latents(work, name, dev)
    mu, sd, y_all, mean, std = lat["mu"], torch.exp(0.5 * lat["logvar"]), lat["y"], lat["mean"], lat["std"]
    model = MLPDenoiser(mu.shape[1]).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    _log(f"flat LDM[{name}_{'cond' if cond else 'uncond'}] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
         f"params, {steps} steps, bs {batch_size}", log_file)
    null, run, t0 = NUM_CLASSES, 0.0, time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(mu), (batch_size,), device=dev)
        z = (mu[idx] + sd[idx] * torch.randn_like(sd[idx]) - mean) / std
        y = (torch.where(torch.rand(batch_size, device=dev) < p_uncond, null, y_all[idx]) if cond
             else torch.full((batch_size,), null, device=dev, dtype=torch.long))
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        loss = v_loss(model, z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        d = min(ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        if step % 50 == 0:
            run += loss.item()
        if step % log_every == 0:
            _log(f"flat LDM[{name}_{'cond' if cond else 'uncond'}] step {step}/{steps} "
                 f"loss {run / (log_every / 50):.4f} ({log_every / (time.time() - t0):.0f} it/s)", log_file)
            run, t0 = 0.0, time.time()
    torch.save({"ema": ema.state_dict(), "dim": mu.shape[1], "cond": cond, "steps": steps}, path(work, "ldm", name, cond))
    return ema


def load_ldm(work, name, cond, dev):
    ck = torch.load(path(work, "ldm", name, cond), map_location=dev)
    model = MLPDenoiser(ck["dim"]).to(dev)
    model.load_state_dict(ck["ema"])
    return model.eval().requires_grad_(False)


class FlatSampler:
    """Same interface as concos_ldm.sample.Sampler, for the flat family."""

    def __init__(self, work, name, cond, device=None):
        self.dev = device or get_device()
        self.vae = load_vae(work, name, self.dev)
        self.model = load_ldm(work, name, cond, self.dev)
        lat = load_latents(work, name, self.dev)
        self.mean, self.std, self.dim = lat["mean"], lat["std"], lat["mu"].shape[1]

    @torch.no_grad()
    def sample(self, n=None, labels=None, guidance=1.0, steps=100, seed=0, chunk=5000):
        if labels is None:
            labels = torch.full((n,), NUM_CLASSES, dtype=torch.long)
        labels = labels.to(self.dev)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        zs = []
        for i in range(0, len(labels), chunk):
            y = labels[i:i + chunk]
            z = ddim_sample(self.model, (len(y), self.dim), y, guidance=guidance, steps=steps, device=self.dev,
                            generator=g, autocast=autocast(self.dev))
            zs.append(z * self.std + self.mean)
        z = torch.cat(zs)
        return self.decode(z), z

    @torch.no_grad()
    def decode(self, z, bs=1000):
        return torch.cat([to_unit(self.vae.decode(z[i:i + bs].to(self.dev)).float()).cpu()
                          for i in range(0, len(z), bs)])

"""Training stages: VAE (+/- ConCoS), latent encoding, latent diffusion."""

import copy
import math
import os
import time

import torch
import torch.nn.functional as F

from .common import (NUM_CLASSES, autocast, batches, get_device, load_cifar10, random_flip,
                     seed_all, to_float)
from .diffusion import v_loss
from .loss import concos_loss
from .models import UNet, VAE


def _log(msg, log_file=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def ckpt_path(work, kind, name=None):
    return os.path.join(work, "ckpt", f"{kind}_{name}.pt" if name else f"{kind}.pt")


def train_vae(work, name, concos_weight, epochs=30, batch_size=128, lr=2e-4, kl_weight=1e-5,
              lpips_weight=1.0, seed=0, log_file=None):
    import lpips
    dev = get_device()
    seed_all(seed)
    (x_tr, y_tr), (x_te, y_te) = load_cifar10(work, dev)
    vae = VAE().to(dev)
    perc = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
    opt = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.9))
    total = epochs * (len(x_tr) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    _log(f"VAE[{name}] params {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M  "
         f"lambda={concos_weight} beta={kl_weight} lpips={lpips_weight}", log_file)
    for ep in range(epochs):
        vae.train()
        sums, n = {}, 0
        for idx in batches(len(x_tr), batch_size, dev):
            x, y = random_flip(to_float(x_tr[idx])), y_tr[idx]
            with autocast(dev):
                recon, mu, logvar = vae(x)
                recon = recon.float()
                l1 = (recon - x).abs().mean()
                lp = perc(recon, x).mean() if lpips_weight > 0 else torch.zeros((), device=dev)
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).flatten(1).sum(1).mean()
            cc = concos_loss(mu, y)
            loss = l1 + lpips_weight * lp + kl_weight * kl + concos_weight * cc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("l1", l1), ("lpips", lp), ("kl", kl), ("concos", cc)):
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        psnr = vae_psnr(vae, x_te[:2000])
        _log(f"VAE[{name}] ep {ep + 1}/{epochs} " + " ".join(f"{k} {v / n:.4f}" for k, v in sums.items())
             + f" | test PSNR {psnr:.2f}", log_file)
    os.makedirs(os.path.dirname(ckpt_path(work, "vae", name)), exist_ok=True)
    torch.save({"state": vae.state_dict(), "concos_weight": concos_weight, "kl_weight": kl_weight,
                "lpips_weight": lpips_weight, "epochs": epochs}, ckpt_path(work, "vae", name))
    return vae


@torch.no_grad()
def vae_psnr(vae, x_uint8, batch_size=500):
    vae.eval()
    dev = next(vae.parameters()).device
    mse = []
    for i in range(0, len(x_uint8), batch_size):
        x = to_float(x_uint8[i:i + batch_size])
        mu, _ = vae.encode(x)
        with autocast(dev):
            r = vae.decode(mu).float().clamp(-1, 1)
        mse.append(((r - x) / 2).pow(2).flatten(1).mean(1))
    return float((-10 * torch.log10(torch.cat(mse))).mean())


def load_vae(work, name, dev):
    vae = VAE().to(dev)
    vae.load_state_dict(torch.load(ckpt_path(work, "vae", name), map_location=dev)["state"])
    return vae.eval().requires_grad_(False)


@torch.no_grad()
def encode_latents(work, name, batch_size=1000, log_file=None):
    """Encode train (+ horizontally flipped) and test sets once; store posterior params + normaliser."""
    dev = get_device()
    (x_tr, y_tr), (x_te, y_te) = load_cifar10(work, dev)
    vae = load_vae(work, name, dev)

    def enc(x_uint8, flip=False):
        mus, lvs = [], []
        for i in range(0, len(x_uint8), batch_size):
            x = to_float(x_uint8[i:i + batch_size])
            mu, lv = vae.encode(x.flip(3) if flip else x)
            mus.append(mu)
            lvs.append(lv)
        return torch.cat(mus), torch.cat(lvs)

    mu_a, lv_a = enc(x_tr)
    mu_b, lv_b = enc(x_tr, flip=True)
    mu_te, lv_te = enc(x_te)
    mu, lv = torch.cat([mu_a, mu_b]), torch.cat([lv_a, lv_b])
    # LDM-style normaliser: per-channel mean, one global scale (keeps relative channel/magnitude structure).
    z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    mean = z.mean(dim=(0, 2, 3), keepdim=True)
    std = (z - mean).std()
    _log(f"latents[{name}] shape {tuple(mu.shape[1:])} mean|.| {mean.abs().mean():.3f} std {std:.3f} "
         f"posterior std {torch.exp(0.5 * lv).mean():.4f}", log_file)
    torch.save({"mu": mu.cpu(), "logvar": lv.cpu(), "y": torch.cat([y_tr, y_tr]).cpu(),
                "mu_test": mu_te.cpu(), "logvar_test": lv_te.cpu(), "y_test": y_te.cpu(),
                "mean": mean[0].cpu(), "std": std.cpu()},
               os.path.join(work, "ckpt", f"latents_{name}.pt"))


def load_latents(work, name, dev):
    lat = torch.load(os.path.join(work, "ckpt", f"latents_{name}.pt"), map_location=dev)
    lat["mean"] = lat["mean"][None]
    return lat


def ldm_name(vae_name, cond):
    return f"{vae_name}_{'cond' if cond else 'uncond'}"


def train_ldm(work, vae_name, cond, steps=60_000, batch_size=256, lr=2e-4, p_uncond=0.1,
              ema_decay=0.9999, warmup=1000, seed=0, ckpt_every=10_000, log_every=1000, log_file=None):
    dev = get_device()
    seed_all(seed)
    name = ldm_name(vae_name, cond)
    lat = load_latents(work, vae_name, dev)
    mu, sd, y_all = lat["mu"], torch.exp(0.5 * lat["logvar"]), lat["y"]
    mean, std = lat["mean"], lat["std"]
    model = UNet(in_ch=mu.shape[1], num_classes=NUM_CLASSES).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    path = ckpt_path(work, "ldm", name)
    step = 0
    if os.path.exists(path + ".partial"):
        st = torch.load(path + ".partial", map_location=dev)
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        _log(f"LDM[{name}] resumed at step {step}", log_file)
    _log(f"LDM[{name}] params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M cond={cond} "
         f"steps={steps} bs={batch_size}", log_file)
    null = NUM_CLASSES
    t0, run = time.time(), 0.0
    model.train()
    while step < steps:
        idx = torch.randint(0, len(mu), (batch_size,), device=dev)
        z = ((mu[idx] + sd[idx] * torch.randn_like(sd[idx])) - mean) / std
        if cond:
            y = torch.where(torch.rand(batch_size, device=dev) < p_uncond, null, y_all[idx])
        else:
            y = torch.full((batch_size,), null, device=dev, dtype=torch.long)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        with autocast(dev):
            loss = v_loss(model, z, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        d = min(ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        run += loss.item() if step % 50 == 0 else 0.0
        if step % log_every == 0:
            rate = log_every / max(time.time() - t0, 1e-6)
            _log(f"LDM[{name}] step {step}/{steps} loss {run / (log_every / 50):.4f} ({rate:.1f} it/s)", log_file)
            run, t0 = 0.0, time.time()
        if step % ckpt_every == 0 or step == steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step}, path + ".partial")
    torch.save({"ema": ema.state_dict(), "cond": cond, "vae": vae_name, "steps": steps}, path)
    os.remove(path + ".partial")
    return ema


def load_ldm(work, vae_name, cond, dev):
    model = UNet(num_classes=NUM_CLASSES).to(dev)
    model.load_state_dict(torch.load(ckpt_path(work, "ldm", ldm_name(vae_name, cond)), map_location=dev)["ema"])
    return model.eval().requires_grad_(False)

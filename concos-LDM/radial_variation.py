"""Intra-class variation by rescaling generated latents: decode gamma * z for z ~ LDM(class).

ConCoS claim: the latent's direction carries class and its magnitude carries instance-level variation.
So for a ConCoS VAE, decoding gamma * z should change the image while keeping its class; for a plain VAE
there is no such guarantee.

    python radial_variation.py              # -> outputs/radial/{radial_grid,radial_metrics}.png + metrics.json
    python radial_variation.py --per-class 100 --guidance 3

Scaling happens in the VAE's raw latent space (where ConCoS acted on mu), not the LDM's normalised space.
Per gamma, over per-class x 10 latents from the class-conditional LDM:
  class acc      ResNet-18 (trained on real CIFAR-10) agrees with the requested class
  same as gamma=1    prediction unchanged from the unscaled sample
  LPIPS to gamma=1   how much the image changed (variation)
  FID            realism of the rescaled set (vs. CIFAR-10 train; small n, so compare relatively)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from concos_ldm.common import CLASSES, DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES, get_device
from concos_ldm.metrics import Inception, classify, frechet_distance, gaussian_stats, get_classifier
from concos_ldm.sample import Sampler, decode

GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
NAMES = {"plain": "Plain VAE", "concos": "ConCoS VAE"}
COLORS = {"plain": "#2a78d6", "concos": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fbfbf8", "#1f1f1f", "#555555", "#e6e5e0"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--out", default=os.path.join(DEFAULT_OUT, "radial"))
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = get_device()

    import lpips
    perc = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    clf = get_classifier(args.work)
    inc = Inception(dev)
    ref = torch.load(os.path.join(args.work, "fid_ref_train.pt"))
    mu_r, s_r = gaussian_stats(ref["feats"])

    labels = torch.arange(NUM_CLASSES).repeat_interleave(args.per_class)
    metrics, grids = {}, {}
    for vn in ("plain", "concos"):
        s = Sampler(args.work, vn, cond=True, device=dev)
        _, z = s.sample(labels=labels, guidance=args.guidance, steps=args.ddim_steps, seed=args.seed)
        imgs = {g: decode(s.vae, g * z) for g in GAMMAS}
        pred1 = classify(clf, imgs[1.0])
        m = {}
        for g in GAMMAS:
            pred = classify(clf, imgs[g])
            with torch.no_grad():
                d = torch.cat([perc(imgs[g][i:i + 500].to(dev) * 2 - 1, imgs[1.0][i:i + 500].to(dev) * 2 - 1).flatten()
                               for i in range(0, len(labels), 500)])
            m[str(g)] = {"class_acc": float((pred == labels).float().mean()),
                         "same_as_gamma1": float((pred == pred1).float().mean()),
                         "lpips_to_gamma1": float(d.mean()),
                         "FID": frechet_distance(*gaussian_stats(inc.features(imgs[g])), mu_r, s_r)}
            print(f"{vn} gamma={g:g}: " + " ".join(f"{k} {v:.3f}" for k, v in m[str(g)].items()), flush=True)
        metrics[vn] = m
        idx = torch.arange(NUM_CLASSES) * args.per_class
        grids[vn] = make_grid(torch.stack([imgs[g][idx] for g in GAMMAS], 1).flatten(0, 1),
                              nrow=len(GAMMAS), padding=1, pad_value=1.0)
        del s
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"gammas": GAMMAS, "per_class": args.per_class, "guidance": args.guidance, **metrics}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 9.2), facecolor=SURFACE)
    for ax, vn in zip(axes, ("plain", "concos")):
        ax.imshow(grids[vn].permute(1, 2, 0).numpy(), interpolation="nearest")
        h, w = grids[vn].shape[1] / NUM_CLASSES, grids[vn].shape[2] / len(GAMMAS)
        ax.set_yticks([h * (k + 0.5) for k in range(NUM_CLASSES)], CLASSES, fontsize=9, color=MUTED)
        ax.set_xticks([w * (k + 0.5) for k in range(len(GAMMAS))], [f"{g:g}" for g in GAMMAS], fontsize=9, color=MUTED)
        ax.tick_params(length=0)
        ax.set_xlabel("γ  (decode γ · z)", color=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(NAMES[vn], loc="left", fontsize=13, color=INK)
    fig.suptitle(f"Rescaling generated class latents (conditional LDM, CFG w = {args.guidance:g}, same seed)",
                 fontsize=15, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out, "radial_grid.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    panels = [("class_acc", "class accuracy ↑", "Class kept"),
              ("lpips_to_gamma1", "LPIPS to γ = 1", "How much the image changes"),
              ("FID", "FID ↓", "Realism")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)
    for ax, (k, ylab, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for vn in ("plain", "concos"):
            ys = [metrics[vn][str(g)][k] for g in GAMMAS]
            ax.plot(GAMMAS, ys, "-o", color=COLORS[vn], lw=2, ms=7, mec=SURFACE, mew=2, label=NAMES[vn])
        ax.axvline(1.0, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(GAMMAS, [f"{g:g}" for g in GAMMAS])
        ax.minorticks_off()
        ax.set_xlabel("γ", color=MUTED)
        ax.set_ylabel(ylab, color=MUTED)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "radial_metrics.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

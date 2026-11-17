"""Generate images from all four LDMs and put them side by side.

    python generate.py                      # 10x10 grid per model -> outputs/generated_*.png + generated_all.png
    python generate.py --guidance 2 --per-class 8 --seed 1

Conditional models: one row per class (airplane ... truck), CFG scale --guidance.
Unconditional models: --per-class x 10 free samples.
Every model starts from the same noise (same seed), so grid cells are directly comparable.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from concos_ldm.common import CLASSES, DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES
from concos_ldm.sample import Sampler

SURFACE, INK, MUTED = "#fbfbf8", "#1f1f1f", "#555555"
TITLES = {"plain": "Plain VAE", "concos": "ConCoS VAE"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n = NUM_CLASSES * args.per_class
    grids = {}
    for vn in ("plain", "concos"):
        for cond in (False, True):
            s = Sampler(args.work, vn, cond)
            labels = torch.arange(NUM_CLASSES).repeat_interleave(args.per_class) if cond else None
            imgs, _ = s.sample(n=n, labels=labels, guidance=args.guidance if cond else 1.0,
                               steps=args.ddim_steps, seed=args.seed)
            tag = f"{vn}_{'cond' if cond else 'uncond'}"
            save_image(imgs, os.path.join(args.out, f"generated_{tag}.png"), nrow=args.per_class, padding=1)
            grids[(vn, cond)] = make_grid(imgs, nrow=args.per_class, padding=1, pad_value=1.0)
            print("generated", tag, flush=True)
            del s
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 2, figsize=(14, 14.6), facecolor=SURFACE)
    for r, vn in enumerate(("plain", "concos")):
        for c, cond in enumerate((False, True)):
            ax = axes[r, c]
            ax.imshow(grids[(vn, cond)].permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            kind = f"class-conditional, CFG w = {args.guidance:g}" if cond else "unconditional"
            ax.set_title(f"{TITLES[vn]} · {kind}", fontsize=13, color=INK, loc="left")
            if cond:
                h = grids[(vn, cond)].shape[1] / NUM_CLASSES
                ax.set_yticks([h * (k + 0.5) for k in range(NUM_CLASSES)], CLASSES, fontsize=9, color=MUTED)
                ax.tick_params(length=0)
            else:
                ax.set_yticks([])
    fig.suptitle(f"CIFAR-10 latent diffusion samples (DDIM {args.ddim_steps} steps, seed {args.seed})",
                 fontsize=16, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(args.out, "generated_all.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    main()

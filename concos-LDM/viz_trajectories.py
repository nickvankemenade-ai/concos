"""PCA of DDIM sampling trajectories in latent space: plain vs. ConCoS VAE x unconditional vs. conditional (CFG).

    python viz_trajectories.py                  # 2x2 figure -> outputs/trajectories_pca.{png,svg}
    python viz_trajectories.py --guidance 3 --seed 1
    python viz_trajectories.py --method umap    # UMAP instead of PCA -> outputs/trajectories_umap.{png,svg}
    python viz_trajectories.py --fit-traj       # PCA fitted on train data + that panel's trajectories

Everything lives in the diffusion model's (normalised) latent space, flattened to 256-d.
The projection (PCA or UMAP) is fitted per VAE on the real latents only; trajectories are mapped into it
(UMAP via `transform`, which places off-manifold points such as early noisy x_t near their nearest data).
Faint dots are real latents coloured by true class.

  conditional:   one shared starting noise, one trajectory per class (coloured by requested class)
  unconditional: one trajectory per starting noise (10 seeds); there is no label, so each is
                 coloured by a linear probe (trained on real latents) applied to its endpoint.
Start = dot, end = cross.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from concos_ldm.common import CLASSES, DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES, autocast, get_device
from concos_ldm.diffusion import ddim_sample
from concos_ldm import flat
from concos_ldm import train as spatial

# Colour-blind-checked categorical palette, one colour per CIFAR-10 class.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948", "#00a5c4", "#a0522d"]
SURFACE, INK, MUTED, AXIS = "#fbfbf8", "#1f1f1f", "#555555", "#d9d9d2"
TITLES = {"plain": "Plain VAE", "concos": "ConCoS VAE"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--family", choices=["spatial", "flat"], default="spatial",
                   help="spatial = v1 4x8x8 VAEs + UNet LDMs; flat = v2 64-d VAEs + MLP LDMs")
    p.add_argument("--out", default=None, help="default: outputs/ (spatial) or outputs/flat/ (flat)")
    p.add_argument("--vaes", nargs="+", default=["plain", "concos"])
    p.add_argument("--guidance", type=float, default=3.0, help="CFG scale w for the conditional model")
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--method", choices=["pca", "umap"], default="pca")
    p.add_argument("--points", type=int, default=None,
                   help="real data points to fit/draw (default: 50000 for PCA, 20000 for UMAP)")
    p.add_argument("--fit-traj", action="store_true",
                   help="PCA only: fit on the training latents plus the panel's trajectory points (per panel)")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fit_probe(z, y, iters=200):
    lin = torch.nn.Linear(z.shape[1], NUM_CLASSES).to(z.device)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(z), y) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin.requires_grad_(False)


def style_axes(ax, method):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    name = "PC" if method == "pca" else "UMAP"
    ax.set_xlabel(f"{name} 1", fontsize=10, color=MUTED)
    ax.set_ylabel(f"{name} 2", fontsize=10, color=MUTED)


def main():
    args = parse()
    fam = flat if args.family == "flat" else spatial
    if args.out is None:
        args.out = os.path.join(DEFAULT_OUT, "flat") if args.family == "flat" else DEFAULT_OUT
    if args.points is None:
        args.points = 50_000 if args.method == "pca" else 20_000
    dev = get_device()
    fig, axes = plt.subplots(len(args.vaes), 2, figsize=(15, 7.2 * len(args.vaes)), squeeze=False,
                             facecolor=SURFACE)
    for r, vn in enumerate(args.vaes):
        lat = fam.load_latents(args.work, vn, dev)
        shape = tuple(lat["mu"].shape[1:])
        g = torch.Generator(device=dev).manual_seed(args.seed)   # same noises for every VAE
        shared_noise = torch.randn((1, *shape), device=dev, generator=g).expand(NUM_CLASSES, *shape)
        seed_noise = torch.randn((NUM_CLASSES, *shape), device=dev, generator=g)
        n_tr = len(lat["mu"]) // 2                           # un-flipped half of the train set
        norm = lambda m: ((m - lat["mean"]) / lat["std"]).flatten(1)
        z_tr, y_tr = norm(lat["mu"][:n_tr]), lat["y"][:n_tr]
        z_te, y_te = norm(lat["mu_test"]), lat["y_test"]

        probe = fit_probe(z_tr, y_tr)
        probe_acc = float((probe(z_te).argmax(1) == y_te).float().mean())

        pick = torch.randperm(n_tr, device=dev, generator=g)[:args.points]
        data_y = y_tr[pick].cpu()
        if args.method == "umap":
            import umap
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
                                random_state=args.seed)
            data_xy = torch.from_numpy(reducer.fit_transform(z_tr[pick].cpu().numpy()))
            proj = lambda z: torch.from_numpy(reducer.transform(z.cpu().numpy()))
        else:
            center = z_tr[pick].mean(0)
            _, _, vh = torch.linalg.svd(z_tr[pick] - center, full_matrices=False)
            proj = lambda z: ((z - center) @ vh[:2].T).cpu()
            data_xy = proj(z_tr[pick])

        for c, cond in enumerate((False, True)):
            model = fam.load_ldm(args.work, vn, cond, dev)
            if cond:
                labels, x_T, w = torch.arange(NUM_CLASSES, device=dev), shared_noise, args.guidance
            else:
                labels = torch.full((NUM_CLASSES,), NUM_CLASSES, device=dev, dtype=torch.long)
                x_T, w = seed_noise, 1.0
            _, traj = ddim_sample(model, x_T.shape, labels, guidance=w, steps=args.ddim_steps, device=dev,
                                  autocast=autocast(dev), return_traj=True, x_T=x_T)
            traj = traj.flatten(2)
            colour_y = labels if cond else probe(traj[-1]).argmax(1)
            if args.fit_traj and args.method == "pca":
                both = torch.cat([z_tr, traj.reshape(-1, traj.shape[-1])])
                center = both.mean(0)
                _, _, vh = torch.linalg.svd(both - center, full_matrices=False)
                proj = lambda z, center=center, vh=vh: ((z - center) @ vh[:2].T).cpu()
                data_xy = proj(z_tr[pick])
            xy = proj(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape[0], traj.shape[1], 2)

            ax = axes[r, c]
            style_axes(ax, args.method)
            ax.scatter(data_xy[:, 0], data_xy[:, 1], s=2.2, c=[PALETTE[i] for i in data_y], alpha=0.13,
                       linewidths=0, zorder=1, rasterized=True)
            for b in range(xy.shape[1]):
                col = PALETTE[int(colour_y[b])]
                ax.plot(xy[:, b, 0], xy[:, b, 1], color=col, lw=2.2, alpha=0.95, zorder=3,
                        solid_joinstyle="round", solid_capstyle="round")
                ax.scatter(xy[0, b, 0], xy[0, b, 1], s=26, color=col, alpha=0.85, linewidths=0, zorder=4)
                ax.scatter(xy[-1, b, 0], xy[-1, b, 1], s=70, color=col, marker="x", linewidths=2.2, zorder=5)
            if cond:
                sub = f"Class-conditional, CFG w = {args.guidance:g}: one shared start noise, one trajectory per class"
            else:
                sub = (f"Unconditional: 10 start noises, coloured by linear probe on the endpoint "
                       f"(probe test acc {probe_acc:.0%})")
            title = TITLES.get(vn, vn) + (" (64-d, + Euclidean contrast)" if args.family == "flat" else " (4×8×8)")
            ax.set_title(title, fontsize=14, color=INK, loc="left", pad=22)
            ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=MUTED)
        print(f"{vn}: probe test acc {probe_acc:.3f}", flush=True)

    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[k], markersize=6, label=f"{k}: {n}")
               for k, n in enumerate(CLASSES)]
    handles += [plt.Line2D([], [], marker="o", ls="", color=MUTED, markersize=6, label="start (noise)"),
                plt.Line2D([], [], marker="x", ls="", color=MUTED, markersize=8, mew=2, label="end (x₀)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
    space = ("PCA of train data + trajectories" if args.fit_traj and args.method == "pca"
             else f"{args.method.upper()} of train data")
    fig.suptitle(f"Latent flow structure: DDIM trajectories in {space}", fontsize=18, color=INK,
                 x=0.01, y=0.985, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=3)
    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "svg"):
        tag = args.method + ("_fittraj" if args.fit_traj and args.method == "pca" else "")
        path = os.path.join(args.out, f"trajectories_{tag}.{ext}")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        print("wrote", path)


if __name__ == "__main__":
    main()

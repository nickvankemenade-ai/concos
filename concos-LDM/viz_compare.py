"""PCA and UMAP side by side, for every model of a family.

    python viz_compare.py                       # both families -> outputs/compare_{flat,spatial}.png/.svg
    python viz_compare.py --families flat --points 10000
    python viz_compare.py --split test          # project held-out test latents instead (probe still fit on train)

Rows: VAE x {unconditional, conditional}. Columns: PCA | UMAP. Both projections are fitted on the same
real training latents (--points of them), and each model's trajectories are computed once and mapped
into both. Conditional: one shared start noise, one trajectory per class (CFG w). Unconditional: 10 start
noises, coloured by a linear probe on the endpoint. UMAP places off-manifold points (early, noisy x_t)
near their nearest data, so early trajectory segments in the UMAP column are not faithful.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from concos_ldm import flat
from concos_ldm import train as spatial
from concos_ldm.common import CLASSES, DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES, autocast, get_device
from concos_ldm.diffusion import ddim_sample
from viz_trajectories import INK, MUTED, PALETTE, SURFACE, fit_probe, style_axes

FAMILIES = {"flat": (flat, "64-d VAEs, MLP LDMs"),
            "spatial": (spatial, "4×8×8 VAEs, UNet LDMs")}
NAMES = {"spatial": {"plain": "Plain VAE", "concos": "ConCoS VAE"},
         "flat": {"plain": "Euclidean-only VAE", "concos": "ConCoS + Euclidean VAE", "concosonly": "ConCoS-only VAE"}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--families", nargs="+", default=["flat", "spatial"], choices=list(FAMILIES))
    p.add_argument("--vaes", nargs="+", default=["plain", "concos"],
                   help="flat family also has concosonly (ConCoS without the Euclidean term)")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    p.add_argument("--points", type=int, default=10_000)
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="which latents to draw and fit PCA/UMAP on")
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = get_device()
    import umap

    for fam_name in args.families:
        fam, desc = FAMILIES[fam_name]
        rows = [(vn, cond) for vn in args.vaes for cond in (False, True)]
        fig, axes = plt.subplots(len(rows), 2, figsize=(14, 6.4 * len(rows)), facecolor=SURFACE)
        for vi, vn in enumerate(args.vaes):
            lat = fam.load_latents(args.work, vn, dev)
            shape = tuple(lat["mu"].shape[1:])
            n_tr = len(lat["mu"]) // 2
            norm = lambda m: ((m - lat["mean"]) / lat["std"]).flatten(1)
            z_tr, y_tr = norm(lat["mu"][:n_tr]), lat["y"][:n_tr]
            probe = fit_probe(z_tr, y_tr)
            probe_acc = float((probe(norm(lat["mu_test"])).argmax(1) == lat["y_test"]).float().mean())

            g = torch.Generator(device=dev).manual_seed(args.seed)
            pool, pool_y = (z_tr, y_tr) if args.split == "train" else (norm(lat["mu_test"]), lat["y_test"])
            pick = torch.randperm(len(pool), device=dev, generator=g)[:args.points]
            shared = torch.randn((1, *shape), device=dev, generator=g).expand(NUM_CLASSES, *shape)
            seeds = torch.randn((NUM_CLASSES, *shape), device=dev, generator=g)
            data, data_y = pool[pick], pool_y[pick].cpu()

            center = data.mean(0)
            _, _, vh = torch.linalg.svd(data - center, full_matrices=False)
            pca = lambda z: ((z - center) @ vh[:2].T).cpu()
            reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist, random_state=args.seed)
            umap_xy = torch.from_numpy(reducer.fit_transform(data.cpu().numpy()))
            proj = {"PCA": (pca, pca(data)),
                    "UMAP": (lambda z: torch.from_numpy(reducer.transform(z.cpu().numpy())), umap_xy)}
            print(f"[{fam_name}] {vn}: projections fitted, probe acc {probe_acc:.3f}", flush=True)

            for cond in (False, True):
                model = fam.load_ldm(args.work, vn, cond, dev)
                if cond:
                    labels, x_T, w = torch.arange(NUM_CLASSES, device=dev), shared, args.guidance
                else:
                    labels, x_T, w = torch.full((NUM_CLASSES,), NUM_CLASSES, device=dev, dtype=torch.long), seeds, 1.0
                _, traj = ddim_sample(model, x_T.shape, labels, guidance=w, steps=args.ddim_steps, device=dev,
                                      autocast=autocast(dev), return_traj=True, x_T=x_T)
                traj = traj.flatten(2)
                colour_y = labels if cond else probe(traj[-1]).argmax(1)
                r = rows.index((vn, cond))
                for c, method in enumerate(("PCA", "UMAP")):
                    f, data_xy = proj[method]
                    xy = f(traj.reshape(-1, traj.shape[-1])).reshape(traj.shape[0], traj.shape[1], 2)
                    ax = axes[r, c]
                    style_axes(ax, method.lower())
                    ax.scatter(data_xy[:, 0], data_xy[:, 1], s=2.2, c=[PALETTE[i] for i in data_y], alpha=0.15,
                               linewidths=0, zorder=1, rasterized=True)
                    for b in range(xy.shape[1]):
                        col = PALETTE[int(colour_y[b])]
                        ax.plot(xy[:, b, 0], xy[:, b, 1], color=col, lw=2, alpha=0.95, zorder=3,
                                solid_joinstyle="round", solid_capstyle="round")
                        ax.scatter(xy[0, b, 0], xy[0, b, 1], s=24, color=col, alpha=0.85, linewidths=0, zorder=4)
                        ax.scatter(xy[-1, b, 0], xy[-1, b, 1], s=64, color=col, marker="x", linewidths=2.2, zorder=5)
                    kind = (f"conditional, CFG w = {args.guidance:g}, shared start noise" if cond
                            else f"unconditional, 10 noises, coloured by probe ({probe_acc:.0%} acc)")
                    ax.set_title(f"{NAMES[fam_name].get(vn, vn)} · {method}", fontsize=13, color=INK, loc="left", pad=20)
                    ax.text(0, 1.015, kind, transform=ax.transAxes, fontsize=9, color=MUTED)
                print(f"[{fam_name}] {vn} {'cond' if cond else 'uncond'}: done", flush=True)

        handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[k], markersize=6, label=f"{k}: {n}")
                   for k, n in enumerate(CLASSES)]
        handles += [plt.Line2D([], [], marker="o", ls="", color=MUTED, markersize=6, label="start (noise)"),
                    plt.Line2D([], [], marker="x", ls="", color=MUTED, markersize=8, mew=2, label="end (x₀)")]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10, labelcolor=INK,
                   bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
        n_drawn = min(args.points, 10_000) if args.split == "test" else args.points
        fig.suptitle(f"PCA vs UMAP of {n_drawn:,} {args.split}-set latents, with DDIM trajectories · {desc}",
                     fontsize=16, color=INK, x=0.01, y=0.995, ha="left")
        fig.tight_layout(rect=(0, 0.025, 1, 0.985), h_pad=2.5)
        for ext in ("png", "svg"):
            out = os.path.join(args.out, fam_name if fam_name == "flat" else "v1_spatial", f"compare_pca_umap{'' if args.split == 'train' else '_test'}{args.tag}.{ext}")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, dpi=130, facecolor=SURFACE)
            print("wrote", out, flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main()

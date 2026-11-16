"""Train the 4 model variants per dataset and reproduce Tables 1-3 and Figures 1/7/8.

    python reproduce.py                                  # all 12 models (paper setting)
    python reproduce.py --datasets fashion --epochs 5    # quicker run
"""

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from concos import ConCoSAutoencoder
from concos import data as datasets
from concos import metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")
tf.get_logger().setLevel("ERROR")

VARIANTS = {  # name: (variational, uses ConCoS)
    "AE + ConCoS": (False, True),
    "VAE + ConCoS": (True, True),
    "AE": (False, False),
    "VAE": (True, False),
}
H2_GAMMAS = [0.5, 1.0, 5.0, 10.0]                     # Table 2
H2_CURVE = np.geomspace(0.45, 20.0, 40)               # Figure 8
H3_GAMMAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]          # radial feature distance
SWEEP_GAMMAS = np.geomspace(0.25, 4.0, 9)             # radial-scaling figure
COLORS = {"AE + ConCoS": "#1f6feb", "VAE + ConCoS": "#d1242f", "AE": "#8fb8f5", "VAE": "#f2a0a6"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(datasets.DATASETS), choices=datasets.DATASETS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--concos-weight", type=float, default=0.5, help="lambda")
    p.add_argument("--kl-weight", type=float, default=1e-3, help="beta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default=None,
                   help="optional folder with <dataset>/ subfolders of IDX .gz files (skips download)")
    p.add_argument("--out", default="results")
    return p.parse_args()


def train(variational, use_concos, data, args):
    (x_tr, y_tr), _ = data
    tf.keras.utils.set_random_seed(args.seed)
    model = ConCoSAutoencoder(args.latent_dim, variational,
                              concos_weight=args.concos_weight if use_concos else 0.0,
                              kl_weight=args.kl_weight)
    model.compile(optimizer="adam")
    hist = model.fit(x_tr, y_tr, batch_size=args.batch_size, epochs=args.epochs, verbose=0)
    print("    final " + "  ".join(f"{k}={v[-1]:.4f}" for k, v in hist.history.items()))
    return model


def evaluate(model, data, image_pca):
    (x_tr, y_tr), (x_te, y_te) = data
    enc = lambda x: model.encode(x).numpy() if len(x) <= 4096 else np.concatenate(
        [model.encode(x[i:i + 4096]).numpy() for i in range(0, len(x), 4096)])
    dec = lambda z: model.decoder.predict(z.astype("float32"), batch_size=1024, verbose=0)
    z_tr, z_te = enc(x_tr), enc(x_te)

    res = metrics.angular_organisation(z_te, y_te)
    accs = metrics.radial_invariance(z_tr, y_tr, z_te, y_te, H2_GAMMAS + list(H2_CURVE))
    res["radial_acc"] = dict(zip(map(str, H2_GAMMAS), accs[:len(H2_GAMMAS)].tolist()))
    res["radial_curve"] = accs[len(H2_GAMMAS):].tolist()
    idx = np.random.default_rng(0).choice(len(x_te), 512, replace=False)
    res["radial_feat_dist"] = metrics.radial_feature_distance(dec, z_te[idx], image_pca, H3_GAMMAS, x_te[idx])
    res["mi_r_class"] = metrics.mi_radius_class(z_te, y_te)
    res["_pairs"] = metrics.pairwise_cosines(z_te, y_te)
    return res


def plot_dataset(name, results, models, data, out):
    _, (x_te, y_te) = data
    # Figure 8: accuracy under radial scaling
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for v, r in results.items():
        ax.semilogx(H2_CURVE, r["radial_curve"], lw=2, color=COLORS[v], label=v)
    ax.axvline(1.0, color="0.6", ls="--", lw=1)
    ax.set(xlabel=r"radial scale $\gamma$", ylabel="linear-probe accuracy", title=f"{name}: accuracy under $z \\to \\gamma z$")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(f"{out}/{name}_radial_invariance.png", dpi=150); plt.close(fig)

    # Figure 7: intra/inter-class cosine distributions
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    for ax, (v, r) in zip(axes, results.items()):
        intra, inter = r["_pairs"]
        bins = np.linspace(-1, 1, 60)
        ax.hist(intra, bins, density=True, alpha=0.6, label="intra-class")
        ax.hist(inter, bins, density=True, alpha=0.6, label="inter-class")
        ax.set(title=f"{v}  (gap {r['cos_gap']:.2f})", xlabel="cosine similarity")
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(f"{out}/{name}_cosine_distributions.png", dpi=150); plt.close(fig)

    # Figure 1/9: decode gamma * z for one test image
    i = int(np.flatnonzero(y_te == y_te[0])[3])
    fig, axes = plt.subplots(4, len(SWEEP_GAMMAS), figsize=(1.2 * len(SWEEP_GAMMAS), 5.2))
    for row, v in enumerate(["AE", "AE + ConCoS", "VAE", "VAE + ConCoS"]):
        z = models[v].encode(x_te[i:i + 1]).numpy()
        imgs = models[v].decoder.predict(SWEEP_GAMMAS[:, None] * z, verbose=0)
        for col, img in enumerate(imgs):
            ax = axes[row, col]; ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"γ={SWEEP_GAMMAS[col]:.2f}", fontsize=8)
            if col == 0: ax.set_ylabel(v, fontsize=8)
    fig.suptitle(f"{name}: decoding γ·z (direction fixed, magnitude varied)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/{name}_radial_sweep.png", dpi=150); plt.close(fig)


def write_tables(all_results, out):
    rows1, rows2, rows3 = [], [], []
    for ds, res in all_results.items():
        for v, r in res.items():
            rows1.append(f"| {ds} | {v} | {r['intra_cos']:.3f} | {r['inter_cos']:.3f} | {r['cos_gap']:.3f} | {r['simplex_bound']:.3f} |")
            rows2.append(f"| {ds} | {v} | " + " | ".join(f"{a:.3f}" for a in r["radial_acc"].values()) + " |")
            rows3.append(f"| {ds} | {v} | {r['radial_feat_dist']:.3f} | {r['mi_r_class']:.3f} |")
    md = ["## H1: angular semantic organisation", "",
          "| Dataset | Model | Intra-class cos | Inter-class cos | Gap | Simplex bound −1/(C−1) |",
          "|---|---|---|---|---|---|", *rows1, "",
          "## H2: accuracy under radial scaling z → γz", "",
          "| Dataset | Model | " + " | ".join(f"γ={g:g}" for g in H2_GAMMAS) + " |",
          "|---|---|" + "---|" * len(H2_GAMMAS), *rows2, "",
          "## H3: angular / radial separation", "",
          "| Dataset | Model | Radial feat. dist ↑ | MI(r, class) ↓ |", "|---|---|---|---|", *rows3, ""]
    with open(f"{out}/results.md", "w") as f:
        f.write("\n".join(md))
    clean = {d: {v: {k: x for k, x in r.items() if not k.startswith("_")} for v, r in res.items()}
             for d, res in all_results.items()}
    with open(f"{out}/results.json", "w") as f:
        json.dump(clean, f, indent=2)
    print("\n".join(md))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    all_results = {}
    for name in args.datasets:
        data_dir = os.path.join(args.data_root, name) if args.data_root else None
        data = datasets.load(name, data_dir)
        print(f"\n== {name}: {len(data[0][0]):,} train / {len(data[1][0]):,} test")
        image_pca = metrics.fit_image_pca(data[0][0])
        results, models = {}, {}
        for v, (variational, use_concos) in VARIANTS.items():
            t0 = time.time()
            print(f"  training {v} ...")
            models[v] = train(variational, use_concos, data, args)
            results[v] = evaluate(models[v], data, image_pca)
            print(f"    gap={results[v]['cos_gap']:.3f}  done in {time.time() - t0:.0f}s")
        plot_dataset(name, results, models, data, args.out)
        all_results[name] = results
    write_tables(all_results, args.out)


if __name__ == "__main__":
    main()

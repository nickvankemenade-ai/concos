"""Turn evaluate.py's results.json into results.md and two figures."""

import os

VAES = ("plain", "concos")
NAMES = {"plain": "Plain VAE", "concos": "ConCoS VAE"}
COLORS = {"plain": "#2a78d6", "concos": "#eb6834"}          # categorical slots 1, 2 (validated palette)
SURFACE, INK, MUTED, GRID = "#fbfbf8", "#1f1f1f", "#555555", "#e6e5e0"


def _get(res, vn, cond, w, s, n):
    return res.get(f"{vn}_{'cond' if cond else 'uncond'}_w{w:g}_s{s}_n{n}")


def write_report(res, args):
    n, out = args.n, args.out
    n_te = min(n, 10_000)
    L = ["# Did ConCoS make latent diffusion better?", "",
         f"{n:,} samples per row, seed {args.seed}. FID against the 50k CIFAR-10 train images; precision / recall "
         f"(k = 3) against {n:,} of them. Class accuracy = a ResNet-18 trained on real CIFAR-10 agrees with the "
         f"requested class. One training run per model, so small differences may be noise.", "",
         "## Reference points", "",
         "| | FID ↓ | precision ↑ | recall ↑ | class acc ↑ |", "|---|---|---|---|---|"]
    real = res.get(f"real_test_n{n_te}")
    if real:
        L.append(f"| Real test images (noise floor) | {real['FID']:.2f} | {real['precision']:.3f} | "
                 f"{real['recall']:.3f} | {real['class_acc']:.3f} |")
    for vn in VAES:
        r = res.get(f"{vn}_recon_n{n_te}")
        if r:
            L.append(f"| {NAMES[vn]} reconstructions (LDM ceiling) | {r['FID']:.2f} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['class_acc']:.3f} |")

    L += ["", "## 1. Quality and coverage (100 DDIM steps)", "",
          "| Model | FID ↓ | FID − rFID ↓ | precision ↑ | recall ↑ | class acc ↑ | class-balance KL ↓ |",
          "|---|---|---|---|---|---|---|"]
    for cond, w in ((False, 1.0), (True, 1.0), (True, 3.0)):
        for vn in VAES:
            r, rec = _get(res, vn, cond, w, 100, n), res.get(f"{vn}_recon_n{n_te}")
            if not r:
                continue
            kind = f"conditional, w = {w:g}" if cond else "unconditional"
            gap = f"{r['FID'] - rec['FID']:.2f}" if rec else "–"
            acc = f"{r['class_acc']:.3f}" if "class_acc" in r else "–"
            L.append(f"| {NAMES[vn]}, {kind} | {r['FID']:.2f} | {gap} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{acc} | {r['class_hist_kl']:.4f} |")
    L += ["", "FID − rFID: how far the diffusion model is from what its VAE can reconstruct. "
          "Class-balance KL: KL(predicted class histogram ‖ uniform); 0 = all classes equally often.", "",
          "## 2. Guidance trade-off (conditional, 100 DDIM steps)", "",
          "| w | " + " | ".join(f"{NAMES[v]} FID ↓ | {NAMES[v]} class acc ↑" for v in VAES) + " |",
          "|---|" + "---|---|" * len(VAES)]
    for w in args.guidances:
        cells = []
        for vn in VAES:
            r = _get(res, vn, True, w, 100, n)
            cells += [f"{r['FID']:.2f}", f"{r['class_acc']:.3f}"] if r else ["–", "–"]
        L.append(f"| {w:g} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Sampling steps (FID ↓)", "",
          "| DDIM steps | " + " | ".join(f"{NAMES[v]} uncond | {NAMES[v]} cond w=3" for v in VAES) + " |",
          "|---|" + "---|---|" * len(VAES)]
    for s in args.steps:
        cells = []
        for vn in VAES:
            for cond, w in ((False, 1.0), (True, 3.0)):
                r = _get(res, vn, cond, w, s, n)
                cells.append(f"{r['FID']:.2f}" if r else "–")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Figures: `cfg_tradeoff.png` (check 2), `fid_vs_steps.png` (check 4).", ""]
    with open(os.path.join(out, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    _plots(res, args)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _plots(res, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, out = args.n, args.out
    real = res.get(f"real_test_n{min(n, 10_000)}")

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style(ax)
    for vn in VAES:
        pts = [(w, _get(res, vn, True, w, 100, n)) for w in args.guidances]
        pts = [(w, r) for w, r in pts if r]
        if not pts:
            continue
        xs, ys = [r["FID"] for _, r in pts], [r["class_acc"] for _, r in pts]
        ax.plot(xs, ys, "-", color=COLORS[vn], lw=2, zorder=2)
        ax.scatter(xs, ys, s=64, color=COLORS[vn], edgecolors=SURFACE, linewidths=2, zorder=3, label=NAMES[vn])
        for (w, _), x, y in zip(pts, xs, ys):
            ax.annotate(f"w={w:g}", (x, y), textcoords="offset points", xytext=(7, -3), fontsize=8, color=MUTED)
    if real:
        ax.axhline(real["class_acc"], color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(ax.get_xlim()[1], real["class_acc"], "real test images ", ha="right", va="bottom",
                fontsize=8, color=MUTED)
    ax.set_xlabel("FID ↓", color=MUTED)
    ax.set_ylabel("class accuracy of samples ↑", color=MUTED)
    ax.set_title("Guidance trade-off: up-left is better", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cfg_tradeoff.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=False)
    for ax, (cond, w, title) in zip(axes, ((False, 1.0, "Unconditional"), (True, 3.0, "Conditional, w = 3"))):
        _style(ax)
        for vn in VAES:
            pts = [(s, _get(res, vn, cond, w, s, n)) for s in sorted(args.steps)]
            pts = [(s, r["FID"]) for s, r in pts if r]
            if not pts:
                continue
            ax.plot(*zip(*pts), "-o", color=COLORS[vn], lw=2, ms=7, mec=SURFACE, mew=2, label=NAMES[vn])
            ax.annotate(NAMES[vn], pts[-1], textcoords="offset points", xytext=(6, 0), fontsize=8,
                        color=INK, va="center")
        ax.set_xscale("log")
        ax.set_xticks(sorted(args.steps), [str(s) for s in sorted(args.steps)])
        ax.set_xlabel("DDIM steps", color=MUTED)
        ax.set_ylabel("FID ↓", color=MUTED)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fid_vs_steps.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

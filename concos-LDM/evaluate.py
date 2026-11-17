"""Did ConCoS make latent diffusion better? Sample-quality evaluation of the four LDMs.

    python evaluate.py                    # 1k samples per setting (~10 min) -> outputs/eval/results.{json,md} + figures
    python evaluate.py --n 10000          # tighter numbers (~1.5 h)

Checks (all against CIFAR-10, same seed for every model):
  1. Quality / coverage: FID (vs. train-set Inception stats) + precision / recall (k = 3, vs. n train
     images), for every model. Each VAE's reconstruction FID (rFID, test set) is the ceiling its LDMs
     can reach; real test images give the noise floor.
  2. Guidance trade-off: conditional models swept over CFG w; a ResNet-18 trained on real CIFAR-10
     scores whether each sample shows the requested class.
  4. Sampling steps: FID at 10 / 25 / 100 DDIM steps (unconditional, and conditional at w = 3).
Results are saved after every evaluation; re-running skips entries already in results.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from concos_ldm.common import DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES, get_device, load_cifar10, to_float, to_unit
from concos_ldm.metrics import (Inception, classify, frechet_distance, gaussian_stats, get_classifier,
                                precision_recall)
from concos_ldm.sample import Sampler, decode
from concos_ldm.train import load_vae

VAES = ("plain", "concos")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--out", default=os.path.join(DEFAULT_OUT, "eval"))
    p.add_argument("--n", type=int, default=1_000, help="samples per evaluation (10k is ~1.5 h, 1k ~10 min)")
    p.add_argument("--guidances", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 5.0])
    p.add_argument("--steps", type=int, nargs="+", default=[10, 25, 100])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="fewer settings, for a smoke test")
    return p.parse_args()


def main():
    args = parse()
    if args.quick:
        args.guidances, args.steps = [1.0, 3.0], [10, 100]
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    log_path = os.path.join(args.out, "eval.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

    dev = get_device()
    (x_tr, y_tr), (x_te, y_te) = load_cifar10(args.work, dev)
    inc = Inception(dev)

    ref_path = os.path.join(args.work, "fid_ref_train.pt")
    if os.path.exists(ref_path):
        ref = torch.load(ref_path)
    else:
        log("computing Inception features of the 50k train images")
        feats = inc.features(to_unit(to_float(x_tr)))
        ref = {"feats": feats}
        torch.save(ref, ref_path)
    mu_r, s_r = gaussian_stats(ref["feats"])
    g = torch.Generator().manual_seed(args.seed)
    real_pr = ref["feats"][torch.randperm(len(ref["feats"]), generator=g)[:args.n]]

    clf = get_classifier(args.work, log=log)
    n_te = min(args.n, len(x_te))

    def score(images, labels=None):
        f = inc.features(images)
        m = {"FID": frechet_distance(*gaussian_stats(f), mu_r, s_r), **precision_recall(real_pr, f, device=dev)}
        pred = classify(clf, images)
        hist = torch.bincount(pred, minlength=NUM_CLASSES).float() / len(pred)
        m["class_hist"] = [round(float(h), 4) for h in hist]
        m["class_hist_kl"] = float((hist * torch.log(hist.clamp_min(1e-12) * NUM_CLASSES)).sum())
        if labels is not None:
            m["class_acc"] = float((pred == labels.cpu()).float().mean())
        return m

    key = f"real_test_n{n_te}"
    if key not in res:
        res[key] = score(to_unit(to_float(x_te[:n_te])), y_te[:n_te])
        log(f"{key}: {fmt(res[key])}")
        save()

    for vn in VAES:
        key = f"{vn}_recon_n{n_te}"
        if key not in res:
            vae = load_vae(args.work, vn, dev)
            with torch.no_grad():
                mu = torch.cat([vae.encode(to_float(x_te[i:i + 1000]))[0] for i in range(0, n_te, 1000)])
            res[key] = score(decode(vae, mu), y_te[:n_te])
            log(f"{key}: {fmt(res[key])}")
            save()
            del vae

        jobs = [(False, 1.0, s) for s in args.steps]
        jobs += [(True, w, 100) for w in args.guidances]
        jobs += [(True, 3.0, s) for s in args.steps if s != 100]
        for cond in (False, True):
            todo = [(w, s) for c, w, s in jobs if c == cond
                    and f"{vn}_{'cond' if cond else 'uncond'}_w{w:g}_s{s}_n{args.n}" not in res]
            if not todo:
                continue
            sampler = Sampler(args.work, vn, cond, dev)
            for w, s in todo:
                key = f"{vn}_{'cond' if cond else 'uncond'}_w{w:g}_s{s}_n{args.n}"
                t0 = time.time()
                labels = torch.arange(NUM_CLASSES).repeat_interleave(args.n // NUM_CLASSES) if cond else None
                imgs, _ = sampler.sample(n=args.n, labels=labels, guidance=w, steps=s, seed=args.seed)
                res[key] = score(imgs, labels)
                log(f"{key}: {fmt(res[key])} ({time.time() - t0:.0f}s)")
                save()
            del sampler
            torch.cuda.empty_cache()

    from concos_ldm.report import write_report
    write_report(res, args)
    log("done")


def fmt(m):
    return " ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float))


if __name__ == "__main__":
    main()

"""v2 "flat" family: flat 64-d VAEs (plain = recon + Euclidean contrast; concos = + cosine ConCoS) and MLP LDMs.

    python run_flat.py all                         # 2 VAEs -> latents -> 4 LDMs -> sample grids (~25 min on an RTX 4060 laptop)
    python run_flat.py all --vae-epochs 40 --ldm-steps 20000 --euclid-weight 0.01
    python run_flat.py train --name concosonly --concos-weight 0.5 --euclid-weight 0   # one extra model

Skips anything whose checkpoint already exists. Log: <work>/flat_train.log (outside OneDrive),
sample grids: outputs/flat/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torchvision.utils import save_image

from concos_ldm import flat
from concos_ldm.common import DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES

# name -> (ConCoS weight, Euclidean weight or None = --euclid-weight). "plain" is Euclidean-only.
VAES = {"plain": (0.0, None), "concos": (0.5, None)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["all", "train"], help="all = the default pair; train = one model (--name)")
    p.add_argument("--name", help="train: model name, e.g. concosonly")
    p.add_argument("--concos-weight", type=float, default=0.5, help="train: ConCoS weight")
    p.add_argument("--work", default=DEFAULT_WORK)
    p.add_argument("--out", default=os.path.join(DEFAULT_OUT, "flat"))
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--vae-epochs", type=int, default=40)
    p.add_argument("--euclid-weight", type=float, default=0.01)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--ldm-steps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.work, "flat_train.log")

    vaes = VAES if args.cmd == "all" else {args.name: (args.concos_weight, args.euclid_weight)}
    for name, (lam, ew) in vaes.items():
        if not os.path.exists(flat.path(args.work, "vae", name)):
            flat.train_vae(args.work, name, lam, euclid_weight=args.euclid_weight if ew is None else ew,
                           margin=args.margin,
                           z_dim=args.z_dim, epochs=args.vae_epochs, seed=args.seed, log_file=log)
        if not os.path.exists(flat.path(args.work, "latents", name)):
            flat.encode_latents(args.work, name, log_file=log)
    for name in vaes:
        for cond in (False, True):
            if not os.path.exists(flat.path(args.work, "ldm", name, cond)):
                flat.train_ldm(args.work, name, cond, steps=args.ldm_steps, seed=args.seed, log_file=log)
            s = flat.FlatSampler(args.work, name, cond)
            labels = torch.arange(NUM_CLASSES).repeat_interleave(10) if cond else None
            imgs, _ = s.sample(n=100, labels=labels, guidance=3.0 if cond else 1.0, seed=args.seed)
            out = os.path.join(args.out, f"samples_flat_{name}_{'cond_w3' if cond else 'uncond'}.png")
            save_image(imgs, out, nrow=10, padding=1)
            print("wrote", out, flush=True)
    flat._log("all done", log)


if __name__ == "__main__":
    main()

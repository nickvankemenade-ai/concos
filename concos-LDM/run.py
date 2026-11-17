"""concos-LDM: latent diffusion on CIFAR-10 with a ConCoS VAE vs. a plain VAE.

    python run.py all                      # 2 VAEs -> latents -> 4 LDMs (cond+CFG / uncond) -> sample grids
    python run.py vae --name concos        # one stage at a time
    python run.py latents --name concos
    python run.py ldm --vae concos --cond
    python run.py sample --vae concos --cond --guidance 3

Every stage skips work whose checkpoint already exists (LDM training also resumes from
its last partial checkpoint), so `all` can be re-run after an interruption.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from concos_ldm.common import DEFAULT_OUT, DEFAULT_WORK, NUM_CLASSES
from concos_ldm.train import ckpt_path, encode_latents, ldm_name, train_ldm, train_vae

VAES = {"concos": 0.5, "plain": 0.0}  # name -> ConCoS weight lambda (paper default 0.5)


def add_common(p):
    p.add_argument("--work", default=DEFAULT_WORK, help="data/checkpoint dir (default: %(default)s)")
    p.add_argument("--out", default=DEFAULT_OUT, help="sample grids + logs (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("vae")
    add_common(p)
    p.add_argument("--name", choices=list(VAES), required=True)
    p.add_argument("--concos-weight", type=float, default=None, help="override lambda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--kl-weight", type=float, default=1e-5)
    p.add_argument("--lpips-weight", type=float, default=1.0)

    p = sub.add_parser("latents")
    add_common(p)
    p.add_argument("--name", choices=list(VAES), required=True)

    p = sub.add_parser("ldm")
    add_common(p)
    p.add_argument("--vae", choices=list(VAES), required=True)
    p.add_argument("--cond", action="store_true", help="class-conditional with CFG label dropout")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--p-uncond", type=float, default=0.1)

    p = sub.add_parser("sample")
    add_common(p)
    p.add_argument("--vae", choices=list(VAES), required=True)
    p.add_argument("--cond", action="store_true")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--ddim-steps", type=int, default=100)

    p = sub.add_parser("all")
    add_common(p)
    p.add_argument("--vae-epochs", type=int, default=30)
    p.add_argument("--ldm-steps", type=int, default=60_000)
    p.add_argument("--guidances", type=float, nargs="+", default=[1.0, 3.0])
    return ap.parse_args()


def save_grid(args, vae, cond, guidance, per_class=10, ddim_steps=100):
    from torchvision.utils import save_image
    from concos_ldm.sample import Sampler
    s = Sampler(args.work, vae, cond)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(per_class) if cond else None
    imgs, _ = s.sample(n=NUM_CLASSES * per_class, labels=labels, guidance=guidance, steps=ddim_steps, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"samples_{ldm_name(vae, cond)}" + (f"_w{guidance:g}" if cond else "") + ".png")
    save_image(imgs, path, nrow=per_class, padding=1)
    print("wrote", path)


def save_recon_grid(args, names):
    from torchvision.utils import save_image
    from concos_ldm.common import get_device, load_cifar10, to_float, to_unit
    from concos_ldm.sample import decode
    from concos_ldm.train import load_vae
    dev = get_device()
    _, (x_te, _) = load_cifar10(args.work, dev)
    x = to_float(x_te[:16])
    rows = [to_unit(x).cpu()]
    for n in names:
        vae = load_vae(args.work, n, dev)
        rows.append(decode(vae, vae.encode(x)[0]))
    path = os.path.join(args.out, "reconstructions.png")
    save_image(torch.cat(rows), path, nrow=16, padding=1)
    print("wrote", path, "(rows: real, " + ", ".join(names) + ")")


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, "train.log")
    if args.cmd == "vae":
        lam = VAES[args.name] if args.concos_weight is None else args.concos_weight
        train_vae(args.work, args.name, lam, epochs=args.epochs, kl_weight=args.kl_weight,
                  lpips_weight=args.lpips_weight, seed=args.seed, log_file=log)
    elif args.cmd == "latents":
        encode_latents(args.work, args.name, log_file=log)
    elif args.cmd == "ldm":
        train_ldm(args.work, args.vae, args.cond, steps=args.steps, batch_size=args.batch_size,
                  p_uncond=args.p_uncond, seed=args.seed, log_file=log)
    elif args.cmd == "sample":
        save_grid(args, args.vae, args.cond, args.guidance, args.per_class, args.ddim_steps)
    elif args.cmd == "all":
        for name, lam in VAES.items():
            if not os.path.exists(ckpt_path(args.work, "vae", name)):
                train_vae(args.work, name, lam, epochs=args.vae_epochs, seed=args.seed, log_file=log)
            if not os.path.exists(os.path.join(args.work, "ckpt", f"latents_{name}.pt")):
                encode_latents(args.work, name, log_file=log)
        save_recon_grid(args, list(VAES))
        for name in VAES:
            for cond in (False, True):
                if not os.path.exists(ckpt_path(args.work, "ldm", ldm_name(name, cond))):
                    train_ldm(args.work, name, cond, steps=args.ldm_steps, seed=args.seed, log_file=log)
                for w in (args.guidances if cond else [1.0]):
                    save_grid(args, name, cond, w)


if __name__ == "__main__":
    main()

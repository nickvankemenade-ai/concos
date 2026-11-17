"""Paths, seeding, device and the GPU-resident CIFAR-10 loader."""

import json
import os
import random

import numpy as np
import torch

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Heavy files (dataset, checkpoints, latents) live outside the (OneDrive-synced) repo.
DEFAULT_WORK = os.environ.get("CONCOS_LDM_WORK",
                              os.path.join(os.path.expanduser("~"), ".cache", "concos-ldm"))
DEFAULT_OUT = os.path.join(EXT_ROOT, "outputs")

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = 10


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_cifar10(work_dir: str, device):
    """Return (x_train, y_train), (x_test, y_test) as uint8 NCHW / int64 tensors on `device`."""
    import torchvision
    root = os.environ.get("CONCOS_LDM_DATA", os.path.join(work_dir, "data"))
    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(device)
        y = torch.tensor(ds.targets, dtype=torch.long, device=device)
        out.append((x, y))
    return out


def to_float(x_uint8):
    """uint8 [0, 255] -> float [-1, 1]."""
    return x_uint8.float().div_(127.5).sub_(1.0)


def to_unit(x):
    """float [-1, 1] -> float [0, 1]."""
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


def random_flip(x):
    flip = torch.rand(len(x), device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def batches(n, batch_size, device, shuffle=True, drop_last=True):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    stop = n - batch_size + 1 if drop_last else n
    for i in range(0, stop, batch_size):
        yield idx[i:i + batch_size]

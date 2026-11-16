"""MNIST, Fashion-MNIST and EMNIST Letters, scaled to [0, 1] with shape (N, 28, 28, 1)."""

import gzip
import os

import numpy as np

DATASETS = ("mnist", "fashion", "emnist")

_IDX_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _read_idx(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def _load_idx_dir(data_dir):
    out = []
    for split in ("train", "test"):
        img_f, lbl_f = (os.path.join(data_dir, f) for f in _IDX_FILES[split])
        x = _read_idx(img_f, 16).reshape(-1, 28, 28)
        y = _read_idx(lbl_f, 8)
        out.append((x, y))
    return out


def load(name: str, data_dir: str | None = None, seed: int = 42):
    """Return (x_train, y_train), (x_test, y_test).

    If `data_dir` contains the four standard IDX .gz files they are used directly;
    otherwise the dataset is downloaded via keras / the `emnist` package.
    """
    if data_dir and os.path.exists(os.path.join(data_dir, _IDX_FILES["train"][0])):
        (x_tr, y_tr), (x_te, y_te) = _load_idx_dir(data_dir)
    elif name == "mnist":
        from tensorflow.keras.datasets import mnist
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    elif name == "fashion":
        from tensorflow.keras.datasets import fashion_mnist
        (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    elif name == "emnist":
        from emnist import extract_test_samples, extract_training_samples
        x_tr, y_tr = extract_training_samples("letters")
        x_te, y_te = extract_test_samples("letters")
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")

    if name == "emnist":
        # 60k / 10k random subset to match the scale of the other datasets.
        rng = np.random.default_rng(seed)
        tr = rng.choice(len(x_tr), 60_000, replace=False)
        te = rng.choice(len(x_te), 10_000, replace=False)
        x_tr, y_tr, x_te, y_te = x_tr[tr], y_tr[tr], x_te[te], y_te[te]

    prep = lambda x: (x.astype("float32") / 255.0)[..., None]
    return (prep(x_tr), y_tr.astype("int32")), (prep(x_te), y_te.astype("int32"))

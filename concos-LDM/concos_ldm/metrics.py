"""Sample-quality metrics: FID, precision/recall (Kynkaanniemi et al. 2019) and a CIFAR-10 scoring classifier."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from .common import autocast, batches, get_device, load_cifar10, random_flip, seed_all, to_float


class Inception:
    """pytorch-fid InceptionV3 pool3 (2048-d). Input images in [0, 1]."""

    def __init__(self, device):
        from pytorch_fid.inception import InceptionV3
        self.device = device
        self.net = InceptionV3([3]).eval().to(device)

    @torch.no_grad()
    def features(self, images, batch_size=250):
        out = []
        for i in range(0, len(images), batch_size):
            out.append(self.net(images[i:i + batch_size].to(self.device).float())[0].flatten(1).cpu())
        return torch.cat(out)


def gaussian_stats(feats):
    f = feats.double().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


@torch.no_grad()
def _knn_radii(x, k, chunk=2000):
    r = []
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk], x)
        r.append(d.kthvalue(k + 1, dim=1).values)            # +1: skip the zero self-distance
    return torch.cat(r)


@torch.no_grad()
def _coverage(query, ref, ref_radii, chunk=2000):
    hit = []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], ref)
        hit.append((d <= ref_radii[None]).any(dim=1))
    return float(torch.cat(hit).float().mean())


def precision_recall(real_feats, fake_feats, k=3, device="cuda"):
    """Improved precision/recall: fraction of fakes inside the real k-NN manifold, and vice versa."""
    r, f = real_feats.to(device).float(), fake_feats.to(device).float()
    return {"precision": _coverage(f, r, _knn_radii(r, k)), "recall": _coverage(r, f, _knn_radii(f, k))}


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential()
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet18(nn.Module):
    """CIFAR ResNet-18. Only used to score whether generated images show the requested class."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, c = [], 64
        for co, s in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [_Block(c, co, s), _Block(co, co, 1)]
            c = co
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1))


def _random_crop(x, pad=4):
    b, _, h, w = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    iy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ix = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    rows = (iy[:, None] + torch.arange(h, device=x.device))[:, :, None]
    cols = (ix[:, None] + torch.arange(w, device=x.device))[:, None, :]
    return xp[torch.arange(b, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def classify(model, images01, batch_size=1000):
    """images in [0, 1] -> predicted class ids (CPU)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(images01), batch_size):
        with autocast(dev):
            out.append(model(images01[i:i + batch_size].to(dev) * 2 - 1).float().argmax(1).cpu())
    return torch.cat(out)


def get_classifier(work, epochs=30, log=print):
    """Train once (about 10 min on a laptop GPU), then load from <work>/ckpt/classifier.pt."""
    dev = get_device()
    path = os.path.join(work, "ckpt", "classifier.pt")
    model = ResNet18().to(dev)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=dev))
        return model.eval()
    seed_all(0)
    (x_tr, y_tr), (x_te, y_te) = load_cifar10(work, dev)
    bs = 256
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=epochs * (len(x_tr) // bs), pct_start=0.25)
    for ep in range(epochs):
        model.train()
        for idx in batches(len(x_tr), bs, dev):
            x = _random_crop(random_flip(to_float(x_tr[idx])))
            with autocast(dev):
                loss = F.cross_entropy(model(x), y_tr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            acc = float((classify(model, (to_float(x_te) + 1) / 2) == y_te.cpu()).float().mean())
            log(f"classifier ep {ep + 1}/{epochs} test acc {acc:.4f}")
    torch.save(model.state_dict(), path)
    return model.eval()

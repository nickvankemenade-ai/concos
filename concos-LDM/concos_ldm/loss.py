"""ConCoS loss in PyTorch (port of ../concos/loss.py, paper Eq. 2-3 and 7).

For spatial latents (B, C, H, W) the latent is flattened to a (B, C*H*W) vector,
so ConCoS organises the direction of the *whole* latent tensor by class.
"""

import torch
import torch.nn.functional as F


def concos_from_critic(z: torch.Tensor, critic: torch.Tensor) -> torch.Tensor:
    """Generalised ConCoS (Eq. 7). `critic` is a (B, B) matrix with values in [-1, 1]."""
    u = F.normalize(z.flatten(1).float(), dim=1)
    cos = u @ u.T
    mask = 1.0 - torch.eye(len(u), device=u.device, dtype=u.dtype)
    return -(critic * cos * mask).sum() / mask.sum()


def euclid_contrast(z: torch.Tensor, labels: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Euclidean contrastive loss (Hadsell et al. 2006): pull same-class pairs together, push other-class
    pairs at least `margin` apart. Distances are divided by sqrt(dim) so the margin is size-independent."""
    z = z.flatten(1).float()
    sq = z.pow(2).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * z @ z.T).clamp_min(0.0) / z.shape[1]
    labels = labels.reshape(-1)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos, neg = same & ~eye, ~same
    d = d2.clamp_min(1e-8).sqrt()
    return d2[pos].mean() + F.relu(margin - d[neg]).pow(2).mean()


def concos_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Supervised ConCoS: attract same-class directions, repel the rest."""
    labels = labels.reshape(-1)
    sigma = (labels[:, None] == labels[None, :]).float() * 2.0 - 1.0
    return concos_from_critic(z, sigma)

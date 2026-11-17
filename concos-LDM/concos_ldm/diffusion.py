"""Continuous-time cosine-schedule diffusion with v-prediction, DDIM sampling and CFG.

    x_t = alpha_t x_0 + sigma_t eps,   alpha_t = cos(pi t / 2), sigma_t = sin(pi t / 2)
    v   = alpha_t eps - sigma_t x_0
"""

import math

import torch
import torch.nn.functional as F


def alpha_sigma(t):
    return torch.cos(0.5 * math.pi * t), torch.sin(0.5 * math.pi * t)


def v_loss(model, x0, y):
    t = torch.rand(len(x0), device=x0.device)
    eps = torch.randn_like(x0)
    a, s = (v.view(-1, *[1] * (x0.dim() - 1)) for v in alpha_sigma(t))
    xt = a * x0 + s * eps
    target = a * eps - s * x0
    return F.mse_loss(model(xt, t, y).float(), target)


@torch.no_grad()
def ddim_sample(model, shape, y, guidance=1.0, steps=100, device="cuda", generator=None, autocast=None,
                return_traj=False, x_T=None):
    """Deterministic DDIM. `y` holds class ids or the null id for every sample.

    guidance = w in  v = v_uncond + w (v_cond - v_uncond):
      w = 1 -> plain conditional, w = 0 -> unconditional, w > 1 -> CFG.
    For an unconditional model pass y = null and guidance = 1.
    With return_traj=True also returns the (steps + 1, B, ...) stack of x_t, from pure noise to x_0.
    """
    null = model.num_classes
    x = torch.randn(shape, device=device, generator=generator) if x_T is None else x_T.to(device)
    traj = [x] if return_traj else None
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    use_cfg = guidance != 1.0 and bool((y != null).any())
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        tb = t.expand(len(x))
        with autocast:
            if use_cfg:
                v = model(torch.cat([x, x]), torch.cat([tb, tb]), torch.cat([y, torch.full_like(y, null)])).float()
                vc, vu = v.chunk(2)
                v = vu + guidance * (vc - vu)
            else:
                v = model(x, tb, y).float()
        a_t, s_t = alpha_sigma(t)
        a_s, s_s = alpha_sigma(s)
        x0 = a_t * x - s_t * v
        eps = s_t * x + a_t * v
        x = a_s * x0 + s_s * eps
        if return_traj:
            traj.append(x)
    return (x, torch.stack(traj)) if return_traj else x

'''
Filename: tcr_sf.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    Motion (W, C) → (T, W, C) using Temporal-Contrast Step-Forward (SF).
    Maintain a per-channel baseline; when |x-base| exceeds adaptive threshold,
    emit a spike at a latency mapped from the excess, then update baseline
    toward x by one threshold step (classic SF behavior).

References:
    - Auge et al., "A Survey of Encoding Techniques for Signal Processing in SNNs",
      Neural Processing Letters, 2021 (Temporal Contrast: SF).
'''

from __future__ import annotations
import torch

__all__ = ["encode"]

def _ensure_T(num_steps):
    if not num_steps or num_steps is False: return 10
    if isinstance(num_steps, int) and num_steps > 0: return int(num_steps)
    raise ValueError(f"num_steps must be positive int or False, got {num_steps!r}")

def _minmax01_wc(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    xmin = x.amin(dim=0, keepdim=True)
    xmax = x.amax(dim=0, keepdim=True)
    rng = (xmax - xmin).clamp_min(eps)
    return (x - xmin) / rng

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    T = _ensure_T(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    if x.dim() != 2:
        raise ValueError(f"SF motion expects (W, C), got {tuple(x.shape)}")
    x = _minmax01_wc(x).contiguous()

    W, C = x.shape
    # use initial baseline as first sample (per-channel)
    base = x[0:1, :].clone()  # [1, C]

    # adaptive threshold from early dynamics statistics
    d = torch.zeros_like(x); d[1:] = (x[1:] - x[:-1]).abs()
    p = 0.75
    thr = torch.quantile(d, q=torch.tensor(p, dtype=torch.float32, device=x.device), dim=0, keepdim=True).clamp_min(1e-4)

    spikes = torch.zeros((T, W, C), device=x.device, dtype=x.dtype)

    for w in range(W):
        delta = x[w:w+1, :] - base                     # [1, C]
        excess = torch.relu(delta.abs() - thr)         # [1, C]
        # strength in [0,1] by normalizing with per-channel max observed delta
        maxd = (d.amax(dim=0, keepdim=True) + 1e-8)
        a = (excess / maxd).clamp(0, 1)                # [1, C]
        t_star = torch.round((T - 1) * (1.0 - a)).to(torch.int64).clamp(0, T - 1)  # [1, C]
        m = (a > 0)[0]                                 # [C]
        if m.any():
            spikes[t_star[0, m], w, torch.arange(C, device=x.device)[m]] = 1.0
            # step-forward baseline update only where fired
            base[:, m] = base[:, m] + thr[:, m] * torch.sign(delta[:, m])

        # small leak toward x to avoid event starvation
        base = 0.99 * base + 0.01 * x[w:w+1, :]

    # ensure at least one spike per channel
    if spikes.sum() == 0:
        j = d.argmax(dim=0)
        spikes[0, j, torch.arange(C, device=x.device)] = 1.0

    return spikes

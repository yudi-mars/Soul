'''
Filename: tcr_mw.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    Motion (W, C) → (T, W, C) using Temporal-Contrast Moving-Window (MW).
    Baseline is a moving average over a short window along W; when |x-mean|
    exceeds an adaptive threshold, emit a spike with latency mapped from the
    deviation magnitude.

References:
    - Auge et al., "A Survey of Encoding Techniques for Signal Processing in SNNs",
      Neural Processing Letters, 2021 (Temporal Contrast: MW).
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
        raise ValueError(f"MW motion expects (W, C), got {tuple(x.shape)}")
    x = _minmax01_wc(x).contiguous()

    W, C = x.shape
    win = max(3, min(9, W // 10 or 3))  # small window relative to sequence length
    spikes = torch.zeros((T, W, C), device=x.device, dtype=x.dtype)

    # precompute moving mean (causal)
    mean = torch.zeros_like(x)
    csum = torch.zeros((1, C), device=x.device, dtype=x.dtype)
    for w in range(W):
        csum = csum + x[w:w+1, :]
        if w - win >= 0:
            csum = csum - x[w - win:w - win + 1, :]
        denom = min(w + 1, win)
        mean[w:w+1, :] = csum / float(denom)

    dev = (x - mean).abs()
    # adaptive threshold per channel
    p = 0.75
    thr = torch.quantile(dev, q=torch.tensor(p, dtype=torch.float32, device=x.device), dim=0, keepdim=True).clamp_min(1e-4)
    maxd = dev.amax(dim=0, keepdim=True).clamp_min(1e-8)
    a = torch.relu(dev - thr) / maxd                        # [W, C]
    t_star = torch.round((T - 1) * (1.0 - a)).to(torch.int64).clamp(0, T - 1)

    m = (a > 0)
    if m.any():
        tw = t_star[m]
        w_idx, c_idx = m.nonzero(as_tuple=True)
        spikes[tw, w_idx, c_idx] = 1.0
    else:
        # ensure at least one event
        j = dev.argmax(dim=0)
        spikes[0, j, torch.arange(C, device=x.device)] = 1.0

    return spikes

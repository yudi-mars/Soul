'''
Filename: tcr_tbr.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    Motion (W, C) → (T, W, C) using Temporal-Contrast TBR.
    1) Per-sample/channel min-max normalization along W.
    2) diff = x[w] - x[w-1] along the real time axis W.
    3) Adaptive threshold via percentile of |diff| per channel.
    4) Strength a = ReLU(|diff|-thr)/max(|diff|), map to latency t* (strong→earlier).

References:
    - Auge et al., "A Survey of Encoding Techniques for Signal Processing in SNNs",
      Neural Processing Letters, 2021 (Temporal Contrast: TBR).
'''

from __future__ import annotations
import torch

__all__ = ["encode"]

def _ensure_T(num_steps):
    if not num_steps or num_steps is False: return 10
    if isinstance(num_steps, int) and num_steps > 0: return int(num_steps)
    raise ValueError(f"num_steps must be positive int or False, got {num_steps!r}")

def _minmax01_wc(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    # x: [W, C], normalize along W per channel
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
        raise ValueError(f"TBR motion expects (W, C), got {tuple(x.shape)}")
    x = _minmax01_wc(x).contiguous()  # [W, C]

    W, C = x.shape
    # diff along real time W; pad first step with zeros
    d = torch.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    ad = d.abs()  # [W, C]

    # adaptive threshold per-channel by percentile (more stable than a fixed scalar)
    p = 0.75
    thr = torch.quantile(ad, q=torch.tensor(p, dtype=torch.float32, device=x.device), dim=0, keepdim=True)
    maxad = ad.amax(dim=0, keepdim=True).clamp_min(1e-8)
    a = torch.relu(ad - thr) / maxad  # strength in [0,1]

    # map strength to latency t*: strong→early
    t_star = torch.round((T - 1) * (1.0 - a)).to(torch.int64).clamp(0, T - 1)   # [W, C]

    # build spikes [T, W, C]; one spike per (w,c) if strength>0
    spikes = torch.zeros((T, W, C), device=x.device, dtype=x.dtype)
    mask = (a > 0)
    if mask.any():
        tw = t_star[mask]
        w_idx, c_idx = mask.nonzero(as_tuple=True)
        spikes[tw, w_idx, c_idx] = 1.0
    else:
        # fallback: guarantee at least one spike at t=0 on max-change positions
        idx = ad.argmax(dim=0)  # [C]
        spikes[0, idx, torch.arange(C, device=x.device)] = 1.0

    return spikes

'''
Filename: tcr_tbr.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-20
Description:
    Temporal-Contrast TBR with shape-aware routing:
      - motion/acoustic: (W,C) -> (T,W,C), diff along W
      - vision         : (C,H,W) -> (T,C,H,W), contrast vs local mean

References:
    - Auge et al., "A Survey of Encoding Techniques for Signal Processing in SNNs"
'''
from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = ["encode"]

def _T(num_steps): return int(num_steps) if num_steps and num_steps>0 else 10

def _minmax_wc(x):  # (W,C) per-channel along W
    xmin = x.amin(0, keepdim=True); xmax = x.amax(0, keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-8)

def _minmax_chw(x): # (C,H,W) per-channel over H,W
    xmin = x.amin((1,2), keepdim=True); xmax = x.amax((1,2), keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-8)

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    T = _T(num_steps)
    x = inputs.to(torch.float32).contiguous()

    if x.dim() == 2:  # (W,C)
        W,C = x.shape
        x = _minmax_wc(x)
        d = torch.zeros_like(x); d[1:] = x[1:] - x[:-1]
        ad = d.abs()
        thr = torch.quantile(ad, q=torch.tensor(0.75, device=x.device), dim=0, keepdim=True)
        a = torch.relu(ad - thr) / (ad.amax(0, keepdim=True) + 1e-8)      # [W,C] in [0,1]
        t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
        spk = torch.zeros((T,W,C), device=x.device, dtype=x.dtype)
        m = a>0
        if m.any():
             idx = m.nonzero(as_tuple=True)
             spk[t_star[m], idx[0], idx[1]] = 1.0
        else:
            j = ad.argmax(0); spk[0, j, torch.arange(C, device=x.device)] = 1.0
        return spk

    if x.dim() == 3:  # (C,H,W)
        C,H,W = x.shape
        x = _minmax_chw(x)
        # local mean by avg-pool
        xm = F.avg_pool2d(x.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        dev = (x - xm).abs()
        thr = torch.quantile(dev.view(C,-1), q=torch.tensor(0.75, device=x.device), dim=1, keepdim=True)
        thr = thr.view(C,1,1)
        a = torch.relu(dev - thr) / (dev.amax(dim=(1,2), keepdim=True) + 1e-8)  # (C,H,W)
        t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
        spk = torch.zeros((T,C,H,W), device=x.device, dtype=x.dtype)
        m = a>0
        if m.any():
            ti = t_star[m]; c,h,w = m.nonzero(as_tuple=True)
            spk[ti, c, h, w] = 1.0
        else:
            c = a.view(C,-1).argmax(dim=1); h = c // W; w = c % W
            spk[0, torch.arange(C, device=x.device), h, w] = 1.0
        return spk

    raise ValueError(f"TBR expects (W,C) or (C,H,W), got {tuple(x.shape)}")

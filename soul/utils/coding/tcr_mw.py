'''
Filename: tcr_mw.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-20
Description:
    Temporal-Contrast MW with shape-aware routing:
      - motion/acoustic: (W,C) -> (T,W,C), moving-window mean along W
      - vision         : (C,H,W) -> (T,C,H,W), local mean by avg-pool
'''
from __future__ import annotations
import torch
import torch.nn.functional as F

__all__ = ["encode"]

def _T(n): return int(n) if n and n>0 else 10
def _minmax_wc(x):
    xmin = x.amin(0, keepdim=True); xmax = x.amax(0, keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-8)
def _minmax_chw(x):
    xmin = x.amin((1,2), keepdim=True); xmax = x.amax((1,2), keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-8)

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    T = _T(num_steps)
    x = inputs.to(torch.float32).contiguous()

    if x.dim()==2:  # (W,C)
        W,C = x.shape; x = _minmax_wc(x)
        win = max(3, min(9, W // 10 or 3))
        mean = torch.zeros_like(x)
        csum = torch.zeros((1,C), device=x.device, dtype=x.dtype)
        for w in range(W):
            csum = csum + x[w:w+1,:]
            if w - win >= 0: csum = csum - x[w-win:w-win+1,:]
            mean[w:w+1,:] = csum / float(min(w+1, win))
        dev = (x - mean).abs()
        thr = torch.quantile(dev, q=torch.tensor(0.75, device=x.device), dim=0, keepdim=True).clamp_min(1e-4)
        a = (torch.relu(dev - thr) / (dev.amax(0, keepdim=True) + 1e-8)).clamp(0,1)
        t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
        spk = torch.zeros((T,W,C), device=x.device, dtype=x.dtype)
        m = a>0
        if m.any():
            idx = m.nonzero(as_tuple=True)
            spk[t_star[m], idx[0], idx[1]] = 1.0
        else:
            j = dev.argmax(0); spk[0, j, torch.arange(C, device=x.device)] = 1.0
        return spk

    if x.dim()==3:  # (C,H,W)
        C,H,W = x.shape; x = _minmax_chw(x)
        mean = F.avg_pool2d(x.unsqueeze(0), 5, 1, 2).squeeze(0)
        dev  = (x - mean).abs()
        thr = torch.quantile(dev.view(C,-1), q=torch.tensor(0.75, device=x.device), dim=1, keepdim=True)
        thr = thr.view(C,1,1).clamp_min(1e-4)
        a = (torch.relu(dev - thr) / (dev.amax((1,2), keepdim=True) + 1e-8)).clamp(0,1)
        t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
        spk = torch.zeros((T,C,H,W), device=x.device, dtype=x.dtype)
        m = a>0
        if m.any():
            ti = t_star[m]; c,h,w = m.nonzero(as_tuple=True)
            spk[ti, c, h, w] = 1.0
        else:
            c = a.view(C,-1).argmax(1); h = c // W; w = c % W
            spk[0, torch.arange(C, device=x.device), h, w] = 1.0
        return spk

    raise ValueError(f"MW expects (W,C) or (C,H,W), got {tuple(x.shape)}")

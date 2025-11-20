'''
Filename: tcr_sf.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-20
Description:
    Temporal-Contrast SF with shape-aware routing:
      - motion/acoustic: (W,C) -> (T,W,C), baseline step-forward along W
      - vision         : (C,H,W) -> (T,C,H,W), baseline=local mean + tiny leak
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
        base = x[0:1,:].clone()
        d = torch.zeros_like(x); d[1:] = (x[1:] - x[:-1]).abs()
        thr = torch.quantile(d, q=torch.tensor(0.75, device=x.device), dim=0, keepdim=True).clamp_min(1e-4)
        spk = torch.zeros((T,W,C), device=x.device, dtype=x.dtype)
        for w in range(W):
            delta = x[w:w+1,:] - base
            excess = torch.relu(delta.abs() - thr)
            a = (excess / (d.amax(0, keepdim=True) + 1e-8)).clamp(0,1)
            t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
            m = (a>0)[0]
            if m.any():
                spk[t_star[0,m], w, torch.arange(C, device=x.device)[m]] = 1.0
                base[:,m] = base[:,m] + thr[:,m] * torch.sign(delta[:,m])
            base = 0.99*base + 0.01*x[w:w+1,:]
        if spk.sum()==0:
            j = d.argmax(0); spk[0, j, torch.arange(C, device=x.device)] = 1.0
        return spk

    if x.dim()==3:  # (C,H,W)
        C,H,W = x.shape; x = _minmax_chw(x)
        base = F.avg_pool2d(x.unsqueeze(0), 5, 1, 2).squeeze(0)
        dev = (x - base).abs()
        thr = torch.quantile(dev.view(C,-1), q=torch.tensor(0.75, device=x.device), dim=1, keepdim=True)
        thr = thr.view(C,1,1).clamp_min(1e-4)
        spk = torch.zeros((T,C,H,W), device=x.device, dtype=x.dtype)
        # 一次性分配 TTFS（强->早）
        a = (torch.relu(dev - thr) / (dev.amax((1,2), keepdim=True)+1e-8)).clamp(0,1)
        t_star = torch.round((T-1)*(1-a)).to(torch.int64).clamp(0,T-1)
        m = a>0
        if m.any():
            ti = t_star[m]; c,h,w = m.nonzero(as_tuple=True)
            spk[ti, c, h, w] = 1.0
        else:
            c = a.view(C,-1).argmax(1); h = c // W; w = c % W
            spk[0, torch.arange(C, device=x.device), h, w] = 1.0
        return spk

    raise ValueError(f"SF expects (W,C) or (C,H,W), got {tuple(x.shape)}")

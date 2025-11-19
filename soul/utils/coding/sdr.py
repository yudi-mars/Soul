'''
Filename: sdr.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    SDR-style synchrony: at each time bin activate top-k channels per position.
    Produces sparse, synchronous groups across channels.

References:
    - Auge et al., Survey (Correlation & synchrony; SDR).
'''
from __future__ import annotations
import torch
__all__ = ["encode"]

def _restore(spk, mode, tag):
    if mode=="chw": T,P,C=spk.shape; H,W=tag; return spk.reshape(T,H,W,C).permute(0,3,1,2).contiguous()
    else: T,P,C=spk.shape; W,=tag; return spk.reshape(T,W,C).contiguous()

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False, sparsity=0.1, jitter=False) -> torch.Tensor:
    T = int(num_steps) if num_steps and num_steps>0 else 10
    x = inputs.to(torch.float32).contiguous()
    if x.dim()==3:
        C,H,W=x.shape; pc=x.permute(1,2,0).reshape(H*W,C); mode="chw"; tag=(H,W)
    elif x.dim()==2:
        W,C=x.shape; pc=x.reshape(W,C); mode="wc"; tag=(W,)
    else:
        raise ValueError("sdr expects (C,H,W) or (W,C).")
    pc = torch.clamp(torch.nan_to_num(pc),0,1)
    P,C = pc.shape
    k = max(1, int(round(sparsity * C)))

    # top-k per position
    vals, idx = torch.topk(pc, k=min(k,C), dim=1)
    mask = torch.zeros_like(pc)
    mask.scatter_(1, idx, 1.0)

    spikes = torch.zeros((T, P, C), device=pc.device, dtype=pc.dtype)
    if jitter:
        # random time assignment within T
        tsel = torch.randint(low=0, high=T, size=(P,1), device=pc.device)
        for p in range(P):
            spikes[tsel[p].item(), p] = mask[p]
    else:
        # synchronous groups at fixed timeslices (round-robin)
        for t in range(T):
            spikes[t] = mask if (t % 2)==0 else 0.0
    return _restore(spikes, mode, tag)

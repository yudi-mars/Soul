'''
Filename: binary_par.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    Parallel binary code: encode B bits at t=0..B-1 as simultaneous spikes
    repeated over time windows (here simply emit once at t < B).

References:
    - Auge et al., Survey (Correlation & synchrony – parallel binary).
'''
from __future__ import annotations
import torch
__all__ = ["encode"]

def _restore(spk, mode, tag):
    if mode=="chw": T,P,C=spk.shape; H,W=tag; return spk.reshape(T,H,W,C).permute(0,3,1,2).contiguous()
    else: T,P,C=spk.shape; W,=tag; return spk.reshape(T,W,C).contiguous()

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False, bits=4) -> torch.Tensor:
    T = int(num_steps) if num_steps and num_steps>0 else 8
    B = min(bits, T)
    x = inputs.to(torch.float32).contiguous()
    if x.dim()==3:
        C,H,W=x.shape; pc=x.permute(1,2,0).reshape(H*W,C); mode="chw"; tag=(H,W)
    elif x.dim()==2:
        W,C=x.shape; pc=x.reshape(W,C); mode="wc"; tag=(W,)
    else:
        raise ValueError(f"binary_par expects (C,H,W) or (W,C).")
    pc = torch.clamp(torch.nan_to_num(pc),0,1)
    code = torch.round(pc * (2**B - 1)).to(torch.int64)
    spikes = torch.zeros((T, pc.shape[0], pc.shape[1]), device=pc.device, dtype=pc.dtype)
    for t in range(B):
        spikes[t] = ((code >> t) & 1).to(pc.dtype)
    return _restore(spikes, mode, tag)

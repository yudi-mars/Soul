'''
Filename: isi_latency.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    General ISI/latency encoding: map value to inter-spike interval (ISI)
    and emit periodic spikes with that ISI (no burst grouping).

References:
    - Auge et al., Survey (ISI / latency coding).
'''
from __future__ import annotations
import torch
__all__ = ["encode"]

def _restore(spk, mode, tag):
    if mode=="chw": T,P,C=spk.shape; H,W=tag; return spk.reshape(T,H,W,C).permute(0,3,1,2).contiguous()
    else: T,P,C=spk.shape; W,=tag; return spk.reshape(T,W,C).contiguous()

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False, isi_min=1, isi_max=10, epsilon=1e-7) -> torch.Tensor:
    T = int(num_steps) if num_steps and num_steps>0 else 10
    isi_max = max(isi_max, 1)
    x = inputs.to(torch.float32).contiguous()
    if x.dim()==3:
        C,H,W=x.shape; pc=x.permute(1,2,0).reshape(H*W,C); mode="chw"; tag=(H,W)
    elif x.dim()==2:
        W,C=x.shape; pc=x.reshape(W,C); mode="wc"; tag=(W,)
    else:
        raise ValueError("isi_latency expects (C,H,W) or (W,C).")
    pc = torch.clamp(torch.nan_to_num(pc), 0, 1)

    isi = torch.round(isi_max - (isi_max - isi_min) * pc).to(torch.int64).clamp_(1, max(1, isi_max))
    phases = torch.round((1 - pc) * (isi - 1)).to(torch.int64)  # initial offset
    spikes = torch.zeros((T, pc.shape[0], pc.shape[1]), device=pc.device, dtype=pc.dtype)

    base = torch.arange(T, device=pc.device).unsqueeze(1)  # [T,1]
    for idx in range(pc.shape[0]*pc.shape[1]):
        k = idx
        i = k // pc.shape[1]; j = k % pc.shape[1]
        t0 = phases[i, j].item()
        step = isi[i, j].item()
        ts = torch.arange(t0, T, step, device=pc.device)
        spikes.index_put_((ts, torch.tensor(i, device=pc.device), torch.tensor(j, device=pc.device)),
                          torch.ones_like(ts, dtype=pc.dtype), accumulate=False)
    return _restore(spikes, mode, tag)

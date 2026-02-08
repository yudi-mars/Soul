"""
Filename:
    burst.py

Author:
    Weisong Zhang <zws1415@zju.edu.cn>

Date Created:
    2025-11-13

Description:
    Burst coding encoder. Stronger inputs emit denser spike packets (bursts)
    within T time steps. Each element emits B spikes with an input-dependent
    inter-spike interval (ISI), scheduled deterministically.

References:
    - Guo et al., "Neural Coding in Spiking Neural Networks: A Comparative Study
    for Robust Neuromorphic Systems",Frontiers in Neuroscience, 2021 (burst definition & properties)
    https://github.com/seongsikpark/SNN-neural-coding
"""

from __future__ import annotations
import torch

def _ensure_time_steps(num_steps):
    if not num_steps or num_steps is False:
        return 8
    if isinstance(num_steps, int) and num_steps > 0:
        return int(num_steps)
    raise ValueError(f"num_steps must be a positive int or False, got {num_steps!r}")

def _minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    x_min = x.amin()
    x_max = x.amax()
    rng = x_max - x_min
    if rng <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / rng

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    """Burst coding with time-first output; supports (C,H,W) and (W,C)."""
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()
    x = _minmax01(x)
    
    if x.dim() == 3:
        C, H, W = x.shape
        pc = x.permute(1, 2, 0).reshape(H * W, C) 
        mode = "chw"
    elif x.dim() == 2:
        W, C = x.shape
        pc = x.reshape(W, C)                      
        mode = "wc"
    else:
        raise ValueError(f"Burst coding expects (C,H,W) or (W,C), got {tuple(x.shape)}")

    pc = pc.clamp_(0.0, 1.0)                      
    P, C = pc.shape

    # Burst parameters
    burst_max = min(4, T)                         
    isi_max = max(1, (T - 1) // max(1, burst_max - 1)) if burst_max > 1 else T

    vals = pc.reshape(-1)                         
    B_per = (1 + torch.floor(vals * (burst_max - 1))).to(torch.int64)       
    if isi_max <= 1:
        isi_per = torch.ones_like(B_per)
    else:
        isi_per = torch.round(isi_max - (isi_max - 1) * vals).to(torch.int64).clamp_(1, isi_max)

    spikes_flat = torch.zeros((T, P * C), device=pc.device, dtype=pc.dtype)
    base = torch.arange(P * C, device=pc.device)

    for k in range(burst_max):
        t_k = k * isi_per                     
        valid = (k < B_per) & (t_k < T)
        if valid.any():
            spikes_flat.index_put_(
                (t_k[valid], base[valid]),
                torch.ones_like(t_k[valid], dtype=pc.dtype),
                accumulate=False
            )

    if mode == "chw":
        spikes = spikes_flat.reshape(T, H, W, C).permute(0, 3, 1, 2).contiguous()
    else: 
        spikes = spikes_flat.reshape(T, W, C).contiguous()

    return spikes
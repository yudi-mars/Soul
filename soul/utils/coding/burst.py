'''
Filename: burst.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-13
Description:
    Burst coding encoder. Stronger inputs emit denser spike packets (bursts)
    within T time steps. Each element emits B spikes with an input-dependent
    inter-spike interval (ISI), scheduled deterministically.

References:
    - Guo et al., "Neural Coding in Spiking Neural Networks: A Comparative Study
    for Robust Neuromorphic Systems",Frontiers in Neuroscience, 2021 (burst definition & properties)
    https://github.com/seongsikpark/SNN-neural-coding
'''

from __future__ import annotations
import torch

def _ensure_time_steps(num_steps):
    if not num_steps or num_steps is False:
        return 8
    if isinstance(num_steps, int) and num_steps > 0:
        return int(num_steps)
    raise ValueError(f"num_steps must be a positive int or False, got {num_steps!r}")

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    """Burst coding with time-first output; supports (C,H,W) and (W,C)."""
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()

    if x.dim() == 3:
        # (C, H, W) -> flatten positions to [P, C]
        C, H, W = x.shape
        pc = x.permute(1, 2, 0).reshape(H * W, C)  # [P, C], P=H*W
        mode = "chw"
    elif x.dim() == 2:
        # (W, C) -> [P, C] where P=W
        W, C = x.shape
        pc = x.reshape(W, C)                       # [P, C]
        mode = "wc"
    else:
        raise ValueError(f"Burst coding expects (C,H,W) or (W,C), got {tuple(x.shape)}")

    # Clamp to [0,1]; upstream can normalize if needed.
    pc = pc.clamp_(0.0, 1.0)                       # [P, C]
    P, C = pc.shape

    # Burst parameters
    burst_max = min(4, T)                          # spikes per element in [1, burst_max]
    isi_max = max(1, (T - 1) // max(1, burst_max - 1)) if burst_max > 1 else T

    # Vectorized scheduling on flattened grid
    vals = pc.reshape(-1)                           # [P*C]
    B_per = (1 + torch.floor(vals * (burst_max - 1))).to(torch.int64)        # [P*C], in [1, burst_max]
    if isi_max <= 1:
        isi_per = torch.ones_like(B_per)
    else:
        isi_per = torch.round(isi_max - (isi_max - 1) * vals).to(torch.int64).clamp_(1, isi_max)

    spikes_flat = torch.zeros((T, P * C), device=pc.device, dtype=pc.dtype)
    base = torch.arange(P * C, device=pc.device)

    for k in range(burst_max):
        t_k = k * isi_per                          # [P*C]
        valid = (k < B_per) & (t_k < T)
        if valid.any():
            spikes_flat.index_put_(
                (t_k[valid], base[valid]),
                torch.ones_like(t_k[valid], dtype=pc.dtype),
                accumulate=False
            )

    # Restore to required shape
    if mode == "chw":
        # [T, P*C] -> [T, H, W, C] -> [T, C, H, W]
        spikes = spikes_flat.reshape(T, H, W, C).permute(0, 3, 1, 2).contiguous()
    else:  # "wc"
        # [T, P*C] -> [T, W, C]
        spikes = spikes_flat.reshape(T, W, C).contiguous()

    return spikes
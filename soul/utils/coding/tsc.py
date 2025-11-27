'''
Filename: tsc.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-24
Description:
    Temporal-Switch-Coding (TSC) encoder. Each input element is represented by
    at most two bipolar spikes within T time steps. The inter-spike interval
    (ISI) encodes the magnitude of the input, while the spike signs encode the
    sign of the input.

    For a pixel p ∈ [-1, 1] and N = T time steps:
        p* = floor(|p| * (N - 1)) + 1
    If p* ≥ 2, emit:
        +sgn(p) at t = 1, and -sgn(p) at t = p*
    Otherwise emit no spike.

References:
    - Han et al., "Deep Spiking Neural Network: Energy Efficiency
      Through Time-Based Coding", ECCV, 2020. (TSC definition & properties)
'''

from __future__ import annotations
import torch


def _ensure_time_steps(num_steps):
    """Shared helper: normalize num_steps argument."""
    if not num_steps or num_steps is False:
        return 8
    if isinstance(num_steps, int) and num_steps > 0:
        return int(num_steps)
    raise ValueError(f"num_steps must be a positive int or False, got {num_steps!r}")


@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    """
    Temporal-Switch-Coding (TSC) with time-first output; supports (C,H,W) and (W,C).

    Args:
        inputs: Input tensor. Shape:
            - (C, H, W) for image-like features, or
            - (W, C) for 1D features.
            Values are expected in [-1, 1]; they will be clamped to this range.
        num_steps: Number of time steps N (N >= 2). If False/0, defaults to 8.

    Returns:
        spikes: Time-first spike tensor.
            - If inputs is (C,H,W): shape (T, C, H, W)
            - If inputs is (W,C):   shape (T, W, C)
        Elements are in {-1, 0, +1}, representing bipolar spikes.
    """
    T = _ensure_time_steps(num_steps)
    if T < 2:
        raise ValueError(f"TSC requires num_steps >= 2, got {T}")

    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()

    if x.dim() == 3:
        C, H, W = x.shape
        pc = x.permute(1, 2, 0).reshape(H * W, C) 
        mode = "chw"
    elif x.dim() == 2:
        W, C = x.shape
        pc = x.reshape(W, C)                        
        mode = "wc"
    else:
        raise ValueError(f"TSC coding expects (C,H,W) or (W,C), got {tuple(x.shape)}")

    pc = pc.clamp_(-1.0, 1.0)                    
    P, C = pc.shape

    vals = pc.reshape(-1)

    abs_vals = vals.abs()
    p_star = torch.floor(abs_vals * (T - 1)) + 1.0  

    sign = torch.sign(vals)

    M = vals.numel()
    spikes_flat = torch.zeros((T, M), device=pc.device, dtype=pc.dtype)

    base = torch.arange(M, device=pc.device)

    valid = p_star >= 2.0
    if valid.any():
        idx = base[valid]
        s_valid = sign[valid]                      
        p_star_valid = p_star[valid].to(torch.int64)  

        t1 = torch.zeros_like(idx)
        spikes_flat[t1, idx] = s_valid

        t2 = (p_star_valid - 1).clamp_(0, T - 1)
        spikes_flat[t2, idx] = -s_valid

    if mode == "chw":
        spikes = spikes_flat.reshape(T, H, W, C).permute(0, 3, 1, 2).contiguous()
    else: 
        spikes = spikes_flat.reshape(T, W, C).contiguous()

    return spikes

'''
Filename: poisson.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-13
Description:
    Poisson coding. Normalize inputs to [0,1] inside the encoder, then
    emit a spike at each time step with probability p equal to the normalized value.
    Time-first output with the same spatial shape as inputs.

References:
    snnTorch spikegen.rate (Poisson/binomial spike generator)
    BindsNET PoissonEncoder
    Diehl & Cook (2015), MNIST pixels to Poisson spikes
'''

from __future__ import annotations
import torch


def _ensure_time_steps(num_steps):
    if not num_steps or num_steps is False:
        return 10
    if isinstance(num_steps, int) and num_steps > 0:
        return num_steps
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
    """Poisson rate coding with in-encoder normalization; time-first output [T, ...]."""
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()
    x = _minmax01(x)

    if x.dim() == 3:
        C, H, W = x.shape
        p = x.unsqueeze(0).expand(T, C, H, W)
        return (torch.rand_like(p) < p).to(x.dtype)
    elif x.dim() == 2:
        B, C = x.shape
        p = x.expand(T, B, C)
        return (torch.rand_like(p) < p).to(x.dtype)
    else:
        raise ValueError(f"Poisson coding expects [B,C], [B,C,H,W], [C,H,W], or [W,C], got {tuple(x.shape)}")

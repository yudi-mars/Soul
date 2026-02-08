"""
Filename:
    population.py

Author:
    Weisong Zhang <zws1415@zju.edu.cn>

Date Created:
    2025-11-20

Description:
    Population coding identical in formula and broadcasting semantics to
    Norse's PopulationEncoder: Gaussian tuning curves with uniformly
    spaced centers on [low, high]. This wrapper exposes a two-argument
    interface encode(inputs, num_steps). If num_steps>1, the encoded
    tensor is simply repeated along a new leading time dimension to
    satisfy time-first pipelines without altering the encoding.

References:
    - Norse docs: norse.torch.module.encode.PopulationEncoder
      https://norse.github.io/norse/generated/norse.torch.module.encode.PopulationEncoder.html
    - Norse docs: norse.torch.functional.encode (population encoding)
      https://norse.github.io/norse/auto_api/norse.torch.functional.encode.html
"""

from __future__ import annotations
import torch

__all__ = ["encode"]

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

def population_encode_gaussian(
    x: torch.Tensor,
    out_features: int,
    low: float = 0.0,
    high: float = 1.0,
    beta: float = 100.0,
) -> torch.Tensor:
    """
    Gaussian population encoding (Norse-equivalent).
    Given x of arbitrary shape (...,), returns (..., out_features) where
    y[..., k] = exp(-beta * (x - centers[k])**2), centers are linearly
    spaced on [low, high].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.to(torch.float32)

    centers = torch.linspace(low, high, out_features, device=x.device, dtype=x.dtype)
    # Broadcast to (..., out_features)
    diff2 = (x.unsqueeze(-1) - centers.view(*([1] * x.ndim), -1)) ** 2
    y = torch.exp(-beta * diff2)
    return y

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()
    x = _minmax01(x)

    low, high, beta = 0.0, 1.0, 100.0
    centers = torch.linspace(low, high, T, device=x.device, dtype=x.dtype)

    diff2 = (x.unsqueeze(-1) - centers.view(*([1] * x.ndim), -1)) ** 2
    y = torch.exp(-beta * diff2)  # shape:

    if x.dim() == 3:        # vision: (C,H,W) -> (T,C,H,W)
        C,H,W,_ = y.shape
        return y.permute(3, 0, 1, 2).contiguous()  # (K->T, C, H, W)

    if x.dim() == 2:        # motion: (W,C) -> (T,W,C)
        W,C,_ = y.shape
        return y.permute(2, 0, 1).contiguous()     # (K->T, W, C)

    raise ValueError(f"Population expects (C,H,W) or (W,C), got {tuple(x.shape)}")

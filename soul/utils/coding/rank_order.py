'''
Filename: rank_order.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-11
Description:
    Rank-Order coding. For each sample, sort values along channel axis (or last axis),
    map ranks to discrete time steps, and emit a single spike at the mapped time.

References:
    - Auge et al., "A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks", Neural Processing Letters, 2021
    https://github.com/codingrank
'''
from __future__ import annotations
import torch

def _ensure_time_steps(num_steps: int) -> int:
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive int, got {num_steps!r}")
    return int(num_steps)

def _minmax01(x: torch.Tensor) -> torch.Tensor:
    """
    Global min-max normalization to [0, 1]. (Monotonic; preserves rank order.)
    If x is constant, returns zeros.
    """
    x_min = x.amin()
    x_max = x.amax()
    denom = x_max - x_min
    if torch.isclose(denom, torch.tensor(0.0, device=x.device, dtype=x.dtype)):
        return torch.zeros_like(x)
    return (x - x_min) / (denom + 1e-8)

def _rank_to_time(flat: torch.Tensor, T: int) -> torch.Tensor:
    """
    flat: [N] float32
    returns: t [N] int64 in [0, T-1]
    """
    N = flat.numel()
    if T == 1:
        return torch.zeros((N,), device=flat.device, dtype=torch.long)
    if N == 1:
        return torch.zeros((1,), device=flat.device, dtype=torch.long)

    # Deterministic tie-breaker to avoid massive equal-value collapse.
    # After _minmax01, values are ~[0,1], so eps=1e-6 is safe.
    idx = torch.arange(N, device=flat.device, dtype=torch.float32)
    flat = flat + 1e-6 * (idx / max(N - 1, 1))

    order = torch.argsort(flat, descending=True)                   # [N]
    rank = torch.empty_like(order)
    rank.scatter_(0, order, torch.arange(N, device=flat.device))   # [0..N-1]

    t = (rank * (T - 1)) // (N - 1)                                # [N]
    return t.to(torch.long)

@torch.no_grad()
def encode(inputs, num_steps: int) -> torch.Tensor:
    """
    Only exposes: inputs, num_steps.

    Args:
        inputs: (W, C) or (C, D) or (C, H, W)
        num_steps: T

    Returns:
        spikes: (T, *inputs.shape) float32 in {0, 1}
    """
    # --- your required unified preprocessing ---
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()
    x = _minmax01(x)

    if x.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported inputs.ndim={x.ndim}. Expected 2D or 3D only; got shape={tuple(x.shape)}"
        )

    # Keep sorting stable if any NaN/Inf exist
    if not torch.isfinite(x).all():
        x = x.clone()
        x[~torch.isfinite(x)] = 0.0

    rest = x.shape
    flat = x.reshape(-1)                                           # [N]
    t = _rank_to_time(flat, T)                                     # [N]

    spikes = torch.zeros((T, flat.numel()), device=x.device, dtype=torch.float32)
    spikes[t, torch.arange(flat.numel(), device=x.device)] = 1.0

    return spikes.reshape(T, *rest)

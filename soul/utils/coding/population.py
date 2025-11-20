'''
Filename: population.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-20
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
'''

from __future__ import annotations
import torch

__all__ = ["encode", "population_encode_gaussian"]

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
@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    # ---- check T and cast ----
    T = 1 if (not num_steps or num_steps is False) else int(num_steps)
    if T <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps!r}")
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()

    # ---- Norse-equivalent population responses on last dim (K=T) ----
    # centers in [0,1], beta: inverse variance
    low, high, beta = 0.0, 1.0, 100.0
    centers = torch.linspace(low, high, T, device=x.device, dtype=x.dtype)

    # normalize x to [0,1] per channel/axis to be consistent
    def _minmax(x):
        if x.dim() == 3:  # (C,H,W)
            xmin = x.amin(dim=(1,2), keepdim=True); xmax = x.amax(dim=(1,2), keepdim=True)
        elif x.dim() == 2:  # (W,C)
            xmin = x.amin(dim=0, keepdim=True);     xmax = x.amax(dim=0, keepdim=True)
        else:
            xmin = x.amin(); xmax = x.amax()
        return (x - xmin) / (xmax - xmin + 1e-8)

    x01 = _minmax(x)

    # compute Gaussian responses: last dim = K (=T)
    # y[..., k] = exp(-beta * (x01 - centers[k])^2)
    diff2 = (x01.unsqueeze(-1) - centers.view(*([1] * x01.ndim), -1)) ** 2
    y = torch.exp(-beta * diff2)  # shape:
    # - vision : (C,H,W,K)
    # - motion : (W,C,K)

    # ---- map population dim (K) to TIME dim (T) ----
    if x.dim() == 3:        # vision: (C,H,W) -> (T,C,H,W)
        C,H,W,_ = y.shape
        return y.permute(3, 0, 1, 2).contiguous()  # (K->T, C, H, W)

    if x.dim() == 2:        # motion: (W,C) -> (T,W,C)
        W,C,_ = y.shape
        return y.permute(2, 0, 1).contiguous()     # (K->T, W, C)

    raise ValueError(f"Population expects (C,H,W) or (W,C), got {tuple(x.shape)}")

'''
Filename: population.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-13
Description:
    Population coding using Gaussian tuning curves. Inputs are normalized to [0,1],
    then expanded along K preferred centers via Gaussian responses. The responses
    are treated as per-step spike probabilities to generate a time-first spike train.

References:
    - R.J.B. Zhu et al., "Unsupervised approach to decomposing neural tuning curves",
      Nature Communications, 2023. (tuning curves & Poisson noise modeling)
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
    x_min, x_max = x.amin(), x.amax()
    rng = x_max - x_min
    if rng <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / rng

def _gaussian_population(x01: torch.Tensor, K: int = 8, sigma: float = 0.15) -> torch.Tensor:
    # x01 in [0,1]; expand last "channel" axis into K population channels using Gaussian tuning.
    # centers: linspace(0,1,K); response: exp(-(x-c)^2/(2*sigma^2))
    device = x01.device
    dtype = x01.dtype

    if x01.dim() == 4:    # [B, C, H, W] -> [B, C*K, H, W]
        B, C, H, W = x01.shape
        xv = x01.reshape(B, C, 1, H, W)                 # insert K axis
        centers = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype).reshape(1, 1, K, 1, 1)
        resp = torch.exp(- (xv - centers)**2 / (2.0 * sigma * sigma))  # [B,C,K,H,W]
        return resp.reshape(B, C * K, H, W)

    if x01.dim() == 3:    # [C, H, W] -> [C*K, H, W]
        C, H, W = x01.shape
        xv = x01.reshape(C, 1, H, W)
        centers = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype).reshape(1, K, 1, 1)
        resp = torch.exp(- (xv - centers)**2 / (2.0 * sigma * sigma))  # [C,K,H,W]
        return resp.reshape(C * K, H, W)

    if x01.dim() == 2:    # [B, C] or [W, C] -> [B, C*K]
        B, C = x01.shape
        xv = x01.reshape(B, C, 1)
        centers = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype).reshape(1, 1, K)
        resp = torch.exp(- (xv - centers)**2 / (2.0 * sigma * sigma))  # [B,C,K]
        return resp.reshape(B, C * K)

    raise ValueError(f"Population coding expects [B,C], [B,C,H,W], or [C,H,W], got {tuple(x01.shape)}")

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    """Population coding with Gaussian tuning curves (time-first output)."""
    T = _ensure_time_steps(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x01 = _minmax01(x).clamp_(0.0, 1.0)

    # Expand channels by K via Gaussian tuning
    K = 8
    sigma = 0.15
    p = _gaussian_population(x01, K=K, sigma=sigma).clamp_(0.0, 1.0)

    # Bernoulli sampling per time step (Poisson/binomial approximation)
    if p.dim() == 4:   # [B, C*K, H, W] -> [T, B, C*K, H, W]
        B, CK, H, W = p.shape
        P = p.expand(T, B, CK, H, W)
        return (torch.rand_like(P) < P).to(p.dtype)

    if p.dim() == 3:   # [C*K, H, W] -> [T, C*K, H, W]
        CK, H, W = p.shape
        P = p.unsqueeze(0).expand(T, CK, H, W)
        return (torch.rand_like(P) < P).to(p.dtype)

    if p.dim() == 2:   # [B, C*K] -> [T, B, C*K]
        B, CK = p.shape
        P = p.expand(T, B, CK)
        return (torch.rand_like(P) < P).to(p.dtype)

    raise ValueError(f"Unexpected shape after population expansion: {tuple(p.shape)}")

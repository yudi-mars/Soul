'''
Filename: phase.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-13
Description:
    Binary phase coding (multi-spike) for inputs. It maps normalized values in [0,1]
    to a sequence of spikes across T phases via greedy binary fractional expansion.
 References:
    - Hwang & Kung, "One-Spike SNN: Single-Spike Phase Coding with Base Manipulation
    for ANN-to-SNN Conversion Loss Minimization", IEEE 2025.
'''

from __future__ import annotations
import torch

def _ensure_time_steps(num_steps):
    if not num_steps or num_steps is False:
        return 8
    if isinstance(num_steps, int) and num_steps > 0:
        return num_steps
    raise ValueError(f"num_steps must be a positive int or False, got {num_steps!r}")


def _as_time_first(x: torch.Tensor, T: int) -> torch.Tensor:
    # Expect 2D [W,C], 3D [C,H,W], 4D [B,C,H,W], or 2D [B,C]
    if x.dim() == 2:
        # [W,C] or [B,C] are both acceptable; treat first dim as "batch/positions"
        B = x.shape[0]
        C = x.shape[1]
        residual = x.clamp(0, 1).to(x.dtype)
        spikes = torch.zeros((T, B, C), dtype=x.dtype, device=x.device)
        weights = torch.pow(2.0, -(torch.arange(T, device=x.device, dtype=x.dtype) + 1))  # 2^{-(t+1)}
        for t in range(T):
            thr = weights[t]
            s = (residual >= thr).to(x.dtype)
            spikes[t] = s
            residual = residual - s * thr
        return spikes  # [T, B, C]
    elif x.dim() == 3:
        # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)
        B, C, H, W = x.shape
        residual = x.clamp(0, 1).to(x.dtype)
        spikes = torch.zeros((T, B, C, H, W), dtype=x.dtype, device=x.device)
        weights = torch.pow(2.0, -(torch.arange(T, device=x.device, dtype=x.dtype) + 1))
        for t in range(T):
            thr = weights[t]
            s = (residual >= thr).to(x.dtype)
            spikes[t] = s
            residual = residual - s * thr
        return spikes[:, 0]  # [T, C, H, W]
    elif x.dim() == 4:
        # [B,C,H,W]
        B, C, H, W = x.shape
        residual = x.clamp(0, 1).to(x.dtype)
        spikes = torch.zeros((T, B, C, H, W), dtype=x.dtype, device=x.device)
        weights = torch.pow(2.0, -(torch.arange(T, device=x.device, dtype=x.dtype) + 1))
        for t in range(T):
            thr = weights[t]
            s = (residual >= thr).to(x.dtype)
            spikes[t] = s
            residual = residual - s * thr
        return spikes  # [T, B, C, H, W]
    else:
        raise ValueError(f"Phase coding expects [W,C], [B,C], [C,H,W], or [B,C,H,W], got {tuple(x.shape)}")

def encode(inputs: torch.Tensor, num_steps=False) -> torch.Tensor:
    """
    Binary phase coding for inputs.
    Args:
        inputs: float tensor, normalized or unnormalized (we clamp to [0,1]).
        num_steps: number of phases (T). If False, defaults to 4.
    Returns:
        spikes: time-first tensor with shape [T, ...] matching spatial dims of inputs.
    """
    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs, dtype=torch.float32)
    inputs = inputs.to(dtype=torch.float32)
    T = _ensure_time_steps(num_steps)
    # Normalize to [0,1] using per-tensor min-max if needed; to keep it simple and
    # stable across datasets, we only clamp here. Upstream should scale data properly.
    # If you need automatic min-max scaling, uncomment the following lines:
    # x_min, x_max = inputs.amin(dim=None), inputs.amax(dim=None)
    # if (x_max - x_min) > 0:
    #     inputs = (inputs - x_min) / (x_max - x_min)
    # else:
    #     inputs = torch.zeros_like(inputs)

    inputs = inputs.clamp(0, 1)
    return _as_time_first(inputs, T)

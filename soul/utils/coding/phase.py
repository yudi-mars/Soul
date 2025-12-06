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
    if num_steps is False or num_steps is None:
        return 4
    if isinstance(num_steps, int) and num_steps > 0:
        return num_steps
    raise ValueError(f"num_steps must be a positive int or False/None, got {num_steps!r}")


def _phase_like_encoder(x: torch.Tensor, T: int) -> torch.Tensor:

    x_float = x.to(dtype=torch.float32)
    q = (x_float * 256.0).long()

    out_shape = (T,) + tuple(x.shape)
    outputs = torch.zeros(out_shape, dtype=x_float.dtype, device=x.device)

    val = 1.0
    for i in range(T):
        if i < 8:
            bit_idx = 8 - i - 1
            mask = ((q >> bit_idx) & 1) != 0 
            outputs[i][mask] = val
            val /= 2.0
        else:
            outputs[i] = outputs[i % 8]

    return outputs

def _minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    x_min = x.amin()
    x_max = x.amax()
    rng = x_max - x_min
    if rng <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / rng

def encode(inputs: torch.Tensor, num_steps: int = 4) -> torch.Tensor:

    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = _minmax01(x).clamp_(0.0, 1.0)

    inputs = inputs.to(dtype=torch.float32)

    T = _ensure_time_steps(num_steps)
    return _phase_like_encoder(x, T)

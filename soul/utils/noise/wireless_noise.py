import torch

import random

def add_gaussian_noise(x: torch.Tensor, sigma: float):
    '''
    Additive Gaussian noise applied to CSI amplitude data

    Parameters
    ----------
    x : torch.Tensor
        inputs shape (time_step, B, A, S, T)
    sigma : float
        noise standard deviation
    '''
    noise = torch.randn_like(x) * sigma

    return x + noise

def add_dropouts_noise(x: torch.Tensor, drop_rate: float, fill_value: float=0.0):
    '''
    Simulate packet/frame loss — randomly set certain frames along the T dimension to all zeros or a fixed fill value

    Parameters
    ----------
    x : torch.Tensor
        inputs shape (time_step, B, A, S, T)
    drop_rate : float
        Probability of each frame being dropped
    fill_value : float, optional
        Fill value for dropped frames, by default 0.0
    '''
    time_step, B, A, S, T = x.shape
    out = x.clone()

    mask = (torch.rand(B, T, device=x.device) >= drop_rate)
    # mask shape (B, T); broadcast to (B, A, S, T)
    mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(time_step, B, A, S, T)
    out = out * mask + (~mask) * fill_value

    return out

def add_impulse_noise(x: torch.Tensor, burst_prob: float, amplitude_scale: float=2.):
    '''
    Simulate burst noise — randomly generate one or a few sudden jumps along the time dimension T, and apply amplified perturbations to all or part of the (A, S) channels.

    Parameters
    ----------
    x : torch.Tensor
        inputs shape (time_step, B, A, S, T)
    burst_prob : float
        Probability of a burst occurring for each sample
    amplitude_scale : float, optional
        Step/change amplitude ratio (multiply the amplitude by this ratio), by default 2.
    '''
    time_step, B, A, S, T = x.shape
    for t in range(time_step):
        out = x[t].clone()
        for b in range(B):
            if random.random() < burst_prob:
                # Select the starting point in time
                t0 = random.randint(0, T - 1)
                # Optional duration l
                l = random.randint(1, max(1, T // 10))
                t1 = min(T, t0 + l)
                # Apply perturbation to all (A, S) channels or a selected subset
                sign = 1.0 if random.random() < 0.5 else -1.0
                out[b, :, :, t0:t1] = out[b, :, :, t0:t1] * (1.0 + sign * amplitude_scale)

        x[t] = out

    return out

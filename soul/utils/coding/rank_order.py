'''
Filename: rank_order.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-11
Description:
    Rank-Order coding. For each sample, sort values along channel axis (or last axis),
    map ranks to discrete time steps, and emit a single spike at the mapped time.

References:
    - Alan Jeffares et al., "Spike-inspired rank coding for fast and accurate recurrent neural networks", Proc. ICLR 2022.
    https://github.com/codingrank
'''
import torch

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps: int) -> torch.Tensor:
    '''
    Rank-Order coding across channel dimension.
    For each spatial position, channels are sorted by intensity; ranks map to spike times.
    Args:
        inputs: Tensor [B, C, H, W] or [..., C] where C is the ranked axis.
        num_steps: length T.
    Returns:
        spikes: [T, *inputs.shape] with one spike per (channel, position).
    '''
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive int, got {num_steps!r}")
    T = int(num_steps)

    x = inputs
    if x.dim() == 4:
        # [B, C, H, W]
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, C)         # [B*H*W, C]
        order = torch.argsort(flat, dim=1, descending=True) # [N, C]
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(C, device=x.device).unsqueeze(0).expand_as(order))
        ranks = inv.view(B, H, W, C).permute(0, 3, 1, 2)    # [B, C, H, W]
        if C == 1:
            t_star = torch.zeros_like(ranks, dtype=torch.long)
        else:
            t_star = (ranks.float() * max(T - 1, 0) / (C - 1)).round().long()
        t_star = t_star.clamp_(0, max(T - 1, 0))

        spikes = torch.zeros((T, B, C, H, W), device=x.device, dtype=x.dtype)
        flat_spikes = spikes.view(T, -1)
        idx = t_star.reshape(-1)
        base = torch.arange(B * C * H * W, device=x.device)
        flat_spikes.index_put_((idx, base), torch.ones_like(idx, dtype=x.dtype), accumulate=False)
        return flat_spikes.view(T, B, C, H, W)

    elif x.dim() == 3:
        # [C, H, W] -> treat as batch size 1
        x4 = x.unsqueeze(0)  # [1, C, H, W]
        B, C, H, W = x4.shape
        flat = x4.permute(0, 2, 3, 1).reshape(-1, C)
        order = torch.argsort(flat, dim=1, descending=True)
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(C, device=x.device).unsqueeze(0).expand_as(order))
        ranks = inv.view(B, H, W, C).permute(0, 3, 1, 2)  # [1, C, H, W]
        if C == 1:
            t_star = torch.zeros_like(ranks, dtype=torch.long)
        else:
            t_star = (ranks.float() * max(T - 1, 0) / (C - 1)).round().long()
        t_star = t_star.clamp_(0, max(T - 1, 0))

        spikes = torch.zeros((T, B, C, H, W), device=x.device, dtype=x.dtype)
        flat_spikes = spikes.view(T, -1)
        idx = t_star.reshape(-1)
        base = torch.arange(B * C * H * W, device=x.device)
        flat_spikes.index_put_((idx, base), torch.ones_like(idx, dtype=x.dtype), accumulate=False)
        return flat_spikes.view(T, B, C, H, W)[:, 0]

    elif x.dim() == 2:
        # [B, C]
        B, C = x.shape
        order = torch.argsort(x, dim=1, descending=True)
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(C, device=x.device).unsqueeze(0).expand_as(order))
        if C == 1:
            t_star = torch.zeros_like(inv, dtype=torch.long)
        else:
            t_star = (inv.float() * max(T - 1, 0) / (C - 1)).round().long()
        t_star = t_star.clamp_(0, max(T - 1, 0))

        spikes = torch.zeros((T, B, C), device=x.device, dtype=x.dtype)
        flat_spikes = spikes.view(T, -1)
        idx = t_star.reshape(-1)
        base = torch.arange(B * C, device=x.device)
        flat_spikes.index_put_((idx, base), torch.ones_like(idx, dtype=x.dtype), accumulate=False)
        return flat_spikes.view(T, B, C)

    else:
        raise ValueError(f"Rank-Order expects [B,C,H,W] or [B,C], got {tuple(x.shape)}")
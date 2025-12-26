'''
Filename: bsa.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-20
Description:
    Ben's Spiker Algorithm with shape-aware routing:
      - vision:  (C,H,W) -> (T,C,H,W)
      - motion:  (W,C)   -> (T,W,C)
    Per-channel min-max normalization; causal kernel; greedy residue update;
    strength -> latency mapping (stronger -> earlier).

References:
    - Schrauwen & Van Campenhout, IJCNN 2003 (BSA)
    - Auge et al., Neural Processing Letters 2021 (Survey)
'''
from __future__ import annotations
import torch

__all__ = ["encode"]

def _ensure_T(num_steps):
    if not num_steps or num_steps is False: return 20
    if isinstance(num_steps, int) and num_steps > 0: return int(num_steps)
    raise ValueError(f"num_steps must be positive int or False, got {num_steps!r}")

def _minmax01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.to(torch.float32)
    x_min = x.amin()
    x_max = x.amax()
    rng = x_max - x_min
    if rng <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / rng

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False, kernel_len=9) -> torch.Tensor:
    T = _ensure_T(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    x = x.to(torch.float32).contiguous()
    x = _minmax01(x)
    if x.dim() == 2:
        # acoustic: (W, C) -> (T, W, C)
        W, C = x.shape

        klen = min(max(3, int(kernel_len)), W)
        t = torch.arange(klen, device=x.device, dtype=x.dtype)
        ker = torch.exp(-t / (0.3 * klen))
        ker = ker / (ker.sum() + 1e-8)

        # correlation history (for threshold init)
        corr_hist = []
        for w in range(max(1, W - klen + 1)):
            L = min(klen, W - w)
            seg = x[w:w + L, :]
            corr_hist.append((seg * ker[:L].unsqueeze(1)).sum(dim=0))
        corr_hist = torch.stack(corr_hist, dim=0)

        thr_q = 0.65
        thr = torch.quantile(
            corr_hist,
            q=torch.tensor(float(thr_q), device=x.device),
            dim=0
        ).clamp_min(1e-6)

        cmax = corr_hist.amax(dim=0).clamp_min(1e-6)
        thr = torch.minimum(thr, 0.95 * cmax)

        # fired-position ratio target (independent of T)
        target_lo, target_hi = 0.10, 0.35

        max_trials = 4
        best_spikes = None
        best_gap = float("inf")

        for _ in range(max_trials):
            resid = x.clone()
            spikes = torch.zeros((T, W, C), device=x.device, dtype=torch.float32)

            for w in range(W):
                L = min(klen, W - w)
                seg = resid[w:w + L, :]
                corr = (seg * ker[:L].unsqueeze(1)).sum(dim=0)
                fire = corr >= thr

                if fire.any():
                    a = ((corr - thr).clamp_min(0.0) / (cmax - thr + 1e-8)).clamp(0.0, 1.0)
                    t_star = torch.floor((T - 1) * (1.0 - a)).to(torch.int64).clamp(0, T - 1)

                    idx = torch.arange(C, device=x.device)[fire]
                    spikes[t_star[fire], w, idx] = 1.0

                    seg[:, fire] = (seg[:, fire] - ker[:L].unsqueeze(1)).clamp(0.0, 1.0)
                    resid[w:w + L, :] = seg

            fired = float(spikes.sum().item()) / (W * C + 1e-8)
            gap = 0.0 if (target_lo <= fired <= target_hi) else min(abs(fired - target_lo), abs(fired - target_hi))
            if gap < best_gap:
                best_gap = gap
                best_spikes = spikes

            if fired < target_lo:
                thr = (thr * 0.90).clamp_min(1e-6)
            elif fired > target_hi:
                thr = torch.minimum(thr * 1.10, 0.99 * cmax)
            else:
                return spikes

        spikes = best_spikes
        if spikes is None or spikes.sum() == 0:
            spikes = torch.zeros((T, W, C), device=x.device, dtype=torch.float32)
            w_star = x.argmax(dim=0)
            spikes[0, w_star, torch.arange(C, device=x.device)] = 1.0
        return spikes

    elif x.dim() == 3:
        # vision/wireless: (C, H, W) -> (T, C, H, W)
        C, H, W = x.shape
        pc = x.permute(1, 2, 0).reshape(H * W, C)  # [P, C]
        P = pc.shape[0]

        rank = torch.argsort(pc, dim=0, descending=True)  # [P, C]
        rank_percent = rank.argsort(dim=0).to(torch.float32) / max(1, P - 1)
        t_star = torch.floor((T - 1) * (1.0 - rank_percent)).to(torch.int64).clamp(0, T - 1)

        spikes_pc = torch.zeros((T, P, C), device=x.device, dtype=torch.float32)
        pp = torch.arange(P, device=x.device).unsqueeze(1).expand(P, C).reshape(-1)
        cc = torch.arange(C, device=x.device).unsqueeze(0).expand(P, C).reshape(-1)
        tt = t_star.reshape(-1)
        spikes_pc[tt, pp, cc] = 1.0

        return spikes_pc.reshape(T, H, W, C).permute(0, 3, 1, 2).contiguous()

    else:
        raise ValueError(f"BSA expects (W,C) or (C,H,W), got {tuple(x.shape)}")
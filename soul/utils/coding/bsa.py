'''
Filename: bsa.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-19
Description:
    Ben's Spiker Algorithm (BSA) for motion data.
    Input : (W, C)  — real time length W, channels C
    Output: (T, W, C) — time-first micro-time T per real time step W

    Pipeline:
      1) Per-channel min–max normalization along W to [0, 1].
      2) Build a causal 1D kernel (length klen) shared by all channels.
      3) Greedy over W: compute correlation of residual with kernel; when it
         exceeds a per-channel adaptive threshold, emit one spike at (w, c),
         subtract the kernel footprint from the residual, and continue.
      4) Map correlation strength to latency in [0, T-1] (strong → earlier).

References:
    - Schrauwen & Van Campenhout, “BSA, a fast and accurate spike train encoding
      scheme,” IJCNN 2003.
    - Auge et al., “A Survey of Encoding Techniques for Signal Processing in SNNs,”
      Neural Processing Letters, 2021 (Filter/optimizer-based encoders).
'''
from __future__ import annotations
import torch

__all__ = ["encode"]

def _ensure_T(num_steps):
    if not num_steps or num_steps is False:
        return 20
    if isinstance(num_steps, int) and num_steps > 0:
        return int(num_steps)
    raise ValueError(f"num_steps must be a positive int or False, got {num_steps!r}")

def _minmax01_wc(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    # x: [W, C], normalize along W per channel
    x = x.to(torch.float32)
    xmin = x.amin(dim=0, keepdim=True)
    xmax = x.amax(dim=0, keepdim=True)
    rng = (xmax - xmin).clamp_min(eps)
    return (x - xmin) / rng

@torch.no_grad()
def encode(inputs: torch.Tensor, num_steps=False, kernel_len=9, thr_q=0.85) -> torch.Tensor:
    """
    :param inputs: (W, C) motion features over real time W.
    :param num_steps: micro-time depth T of the encoded spikes.
    :param kernel_len: causal kernel length (along W).
    :param thr_q: per-channel quantile for adaptive threshold on correlation.
    """
    T = _ensure_T(num_steps)
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    if x.dim() != 2:
        raise ValueError(f"BSA motion expects (W, C), got {tuple(x.shape)}")
    x = _minmax01_wc(x).contiguous()

    W, C = x.shape
    klen = int(max(3, min(kernel_len, W)))
    t = torch.arange(klen, device=x.device, dtype=x.dtype)
    ker = torch.exp(-t / (0.3 * klen))
    ker = ker / (ker.sum() + 1e-8)                      # [klen], causal, normalized

    # Pre-estimate per-channel correlation stats on the original signal
    # Use a sliding dot with ker over x to get a stable threshold & max
    # (valid region only; simple loop keeps deps minimal)
    corr_hist = []
    for w in range(W - klen + 1):
        seg = x[w:w + klen, :]                          # [klen, C]
        corr_hist.append((seg * ker.unsqueeze(1)).sum(dim=0))  # [C]
    if len(corr_hist) == 0:
        corr_hist = [x.sum(dim=0)]
    corr_hist = torch.stack(corr_hist, dim=0)           # [Nv, C]
    thr = torch.quantile(corr_hist, q=torch.tensor(float(thr_q), device=x.device), dim=0)  # [C]
    cmax = corr_hist.amax(dim=0).clamp_min(1e-6)        # [C]

    # Greedy BSA over residual
    resid = x.clone()
    spikes = torch.zeros((T, W, C), device=x.device, dtype=x.dtype)

    for w in range(W):                                   # real time axis
        # local correlation on current residual, truncated at sequence end
        L = min(klen, W - w)
        seg = resid[w:w + L, :]                          # [L, C]
        corr = (seg * ker[:L].unsqueeze(1)).sum(dim=0)   # [C]

        fire = corr >= thr                               # [C] bool
        if fire.any():
            # strength -> latency (strong -> earlier)
            a = ((corr - thr).clamp_min(0.0) / (cmax - thr + 1e-8)).clamp(0.0, 1.0)  # [C]
            t_star = torch.round((T - 1) * (1.0 - a)).to(torch.int64).clamp(0, T - 1)

            idx = torch.arange(C, device=x.device)[fire]
            spikes[t_star[fire], w, idx] = 1.0

            # subtract kernel footprint for fired channels (classic BSA residue update)
            # here we subtract a unit-amplitude kernel; this yields sparse robust events
            kcol = ker[:L].unsqueeze(1)                  # [L,1]
            seg[:, fire] = (seg[:, fire] - kcol).clamp(0.0, 1.0)
            resid[w:w + L, :] = seg

    # ensure at least one spike per channel if sequence is too flat
    if spikes.sum() == 0:
        # pick w* with max energy and fire at t=0
        w_star = x.argmax(dim=0)                         # [C]
        spikes[0, w_star, torch.arange(C, device=x.device)] = 1.0

    return spikes

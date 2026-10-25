"""Batched BLS in PyTorch, a line-by-line port of astropy's `bls.c` fast method (v8.0.1).

Same inputs (t - min(t), y - median(y), unit inverse variance), same binning
(bin = min(duration) / oversample, n_bins = ceil(P / bin) + oversample, index = floor(phase/bin)+1,
wrap-padding of the first `oversample` bins), same cumulative-sum window search and the same
log-likelihood objective 0.5 * ivar_in * (y_out - y_in)^2 subject to y_out >= y_in.
Float64 throughout. Usable for benchmarks only if it reproduces astropy's peak periods
(amendment A2 (iv)).
"""

from __future__ import annotations

import math

import numpy as np

DBL_EPSILON = np.finfo(np.float64).eps


def bls_power(t, y, periods, durations, oversample=10, device=None, max_elems=4e7, return_ivar_in=False):
    """BLS power per period. With return_ivar_in, also the in-transit inverse-variance sum (= number of
    in-transit points, unit weights) of the box selected at each period, chosen as bls.c chooses it:
    durations in order, phases ascending, replaced only by a strictly larger objective."""
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    tt = torch.as_tensor(t - t.min(), dtype=torch.float64, device=dev)
    yy = torch.as_tensor(y - np.median(y), dtype=torch.float64, device=dev)
    N = tt.numel()
    sum_y = float(yy.sum())
    sum_ivar = float(N)
    bin_d = float(np.min(durations)) / oversample
    # C round(): halves away from zero (Python's round() is half-to-even: 62.5 -> 62, C gives 63)
    durs = [int(math.floor(d / bin_d + 0.5)) for d in durations]
    periods = np.asarray(periods, np.float64)
    out = np.full(periods.size, -np.inf)
    out_iin = np.full(periods.size, np.nan)
    batch = max(8, int(max_elems // max(N, 1)))
    ones = torch.ones(N, dtype=torch.float64, device=dev)
    for s in range(0, periods.size, batch):
        P = torch.as_tensor(periods[s:s + batch], dtype=torch.float64, device=dev)
        B = P.numel()
        n_bins = torch.ceil(P / bin_d).long() + oversample                  # (B,)
        width = int(n_bins.max()) + 1
        ph = tt[None, :] - P[:, None] * torch.floor(tt[None, :] / P[:, None])  # wrap_into
        ind = (ph / bin_d).long() + 1                                         # (B, N)
        my = torch.zeros(B, width, dtype=torch.float64, device=dev)
        mi = torch.zeros(B, width, dtype=torch.float64, device=dev)
        my.scatter_add_(1, ind, yy.expand(B, N))
        mi.scatter_add_(1, ind, ones.expand(B, N))
        # wrap-pad: mean[n_bins - oversample + j] = mean[1 + j], j = 0..oversample-1
        dst = (n_bins - oversample)[:, None] + torch.arange(oversample, device=dev)[None, :]
        my.scatter_(1, dst, my[:, 1:oversample + 1].clone())
        mi.scatter_(1, dst, mi[:, 1:oversample + 1].clone())
        cy = torch.cumsum(my, dim=1)
        ci = torch.cumsum(mi, dim=1)
        best = torch.full((B,), -math.inf, dtype=torch.float64, device=dev)
        best_iin = torch.full((B,), math.nan, dtype=torch.float64, device=dev)
        cols = torch.arange(width, device=dev)
        for dur in durs:
            if dur >= width:
                continue
            y_in = cy[:, dur:] - cy[:, :-dur]
            i_in = ci[:, dur:] - ci[:, :-dur]
            y_out = sum_y - y_in
            i_out = sum_ivar - i_in
            n_ok = cols[None, : width - dur] <= (n_bins - dur)[:, None]
            ok = n_ok & (i_in >= DBL_EPSILON) & (i_out >= DBL_EPSILON)
            yin_n = y_in / torch.where(ok, i_in, torch.ones_like(i_in))
            yout_n = y_out / torch.where(ok, i_out, torch.ones_like(i_out))
            obj = 0.5 * i_in * (yout_n - yin_n) ** 2
            obj = torch.where(ok & (yout_n >= yin_n), obj, torch.full_like(obj, -math.inf))
            if return_ivar_in:
                j = torch.argmax(obj, dim=1)  # first maximal index, as the C loop's strict '>' keeps
                m = obj.gather(1, j[:, None])[:, 0]
                upd = m > best
                best = torch.where(upd, m, best)
                best_iin = torch.where(upd, i_in.gather(1, j[:, None])[:, 0], best_iin)
            else:
                best = torch.maximum(best, obj.max(dim=1).values)
        out[s:s + B] = best.cpu().numpy()
        out_iin[s:s + B] = best_iin.cpu().numpy()
    return (out, out_iin) if return_ivar_in else out

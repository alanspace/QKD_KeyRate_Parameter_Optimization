"""optimize_batch over slices of points, so memory stays bounded (main .venv only; imports the physics).

optimize_batch's global stage holds [points, min(n_sobol, 8192)] arrays. Measured on the 32 GB Mac (2026-11-09):
n_sobol 4096 needs ~30 MB per point over a ~5 GB base (1000 points peaked at 33.8 GB and swapped; ~21k points at once
were OOM-killed). n_sobol 2^14 uses 8192-wide blocks, so about twice that per point.
Every point uses the same Sobol set and its own top-k and local solves, so the result does not depend on the slicing
(checked against one unsliced call: max rel diff 4e-13). The last slice is padded with copies of its final point and
trimmed afterwards, so every call has the same shape and nothing recompiles.
"""
import time

import numpy as np

SLICE_4096 = 200      # predicted peak ~11 GB
SLICE_16384 = 100     # predicted peak ~11 GB


def default_slice(n_sobol):
    return SLICE_4096 if n_sobol <= 4096 else SLICE_16384


def optimize_sliced(L, nX, preset, n_sobol=4096, top_k=8, seed=0, slice_=None, tag="cold"):
    """Same arguments and per-point outputs as optimize_batch (array outputs with a leading point axis are
    concatenated). Prints one line per slice: time of this slice, elapsed, remaining estimated from slices 2..
    (slice 1 includes compilation)."""
    from src.qkd_opt.optimize import optimize_batch
    L = np.atleast_1d(np.asarray(L, np.float64)); nX = np.atleast_1d(np.asarray(nX, np.float64))
    slice_ = slice_ or default_slice(n_sobol)
    n = len(L)
    starts = list(range(0, n, slice_))
    pad = starts[-1] + slice_ - n
    Lp = np.concatenate([L, np.repeat(L[-1:], pad)]); nXp = np.concatenate([nX, np.repeat(nX[-1:], pad)])
    outs, t0, first = [], time.perf_counter(), 0.0
    for i, s in enumerate(starts, 1):
        ts = time.perf_counter()
        outs.append(optimize_batch(Lp[s:s + slice_], nXp[s:s + slice_], preset, n_sobol=n_sobol, top_k=top_k, seed=seed))
        el = time.perf_counter() - t0
        if i == 1:
            first, eta = el, "n/a"
        else:
            eta = f"{(el - first) / (i - 1) * (len(starts) - i):.0f}s"
        print(f"[{tag}] slice {i}/{len(starts)} ({min(slice_, n - s)} pts): {time.perf_counter() - ts:.1f}s, "
              f"elapsed {el:.0f}s, est. remaining {eta}", flush=True)
    res = {}
    for k, v in outs[0].items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.shape[0] == slice_:
            res[k] = np.concatenate([np.asarray(o[k]) for o in outs])[:n]
        else:
            res[k] = v
    return res

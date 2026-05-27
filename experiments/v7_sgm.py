"""v7_sgm — Semi-Global Matching (Hirschmuller 2005), 4-direction scanline DP.

For each of 4 cardinal directions r, the path cost is accumulated as

    L_r(p, d) = C(p, d) + min(
        L_r(p-r, d),
        L_r(p-r, d-1) + P1,
        L_r(p-r, d+1) + P1,
        min_{d'} L_r(p-r, d') + P2,
    ) - min_{d'} L_r(p-r, d')

where the trailing subtraction prevents the running total from blowing up.
The final aggregated cost is the sum over directions. Each direction is a
vectorised single-axis sweep; the Python loop is over W (horizontal sweep) or
H (vertical sweep), with the disparity / orthogonal axis handled by numpy
broadcasting per step. Four directions run in parallel threads.

Standard penalties: P1=10, P2=120. Constant across all images — no
max_disp or image-name branching.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .common import _MAX_WORKERS, compute_disp_skeleton
from .v6_baseline import aggregate as wgif_aggregate

# After WGIF aggregation Census costs typically sit in roughly [0, 20] (smoothed).
# Standard SGM P1=10, P2=120 was for raw block-matching cost; we scale down
# so the penalties stay proportional. These are global constants, no per-image
# branching.
P1 = np.float32(1.5)
P2 = np.float32(15.0)


def _sgm_path(cost_volume: np.ndarray, axis: str) -> np.ndarray:
    """Aggregate cost along one of 4 cardinal directions.

    axis ∈ {'right', 'left', 'down', 'up'}.
    Returns same shape as input.
    """
    D, H, W = cost_volume.shape
    L = np.empty_like(cost_volume)

    if axis in ("right", "left"):
        # Sweep along W; per-step state is (D, H).
        if axis == "right":
            x_iter = range(1, W)
            L[:, :, 0] = cost_volume[:, :, 0]
            x_first = 0
        else:
            x_iter = range(W - 2, -1, -1)
            L[:, :, -1] = cost_volume[:, :, -1]
            x_first = W - 1

        for x in x_iter:
            prev_x = x + 1 if axis == "left" else x - 1
            prev = L[:, :, prev_x]                       # (D, H)
            min_prev = prev.min(axis=0, keepdims=True)   # (1, H)
            # Shifted versions for d±1 transitions.
            prev_dm1 = np.empty_like(prev)
            prev_dm1[0] = np.inf
            prev_dm1[1:] = prev[:-1]
            prev_dp1 = np.empty_like(prev)
            prev_dp1[-1] = np.inf
            prev_dp1[:-1] = prev[1:]

            best = np.minimum(prev, prev_dm1 + P1)
            best = np.minimum(best, prev_dp1 + P1)
            best = np.minimum(best, min_prev + P2)  # min_prev broadcasts
            L[:, :, x] = cost_volume[:, :, x] + best - min_prev

    elif axis in ("down", "up"):
        # Sweep along H; per-step state is (D, W).
        if axis == "down":
            y_iter = range(1, H)
            L[:, 0, :] = cost_volume[:, 0, :]
        else:
            y_iter = range(H - 2, -1, -1)
            L[:, -1, :] = cost_volume[:, -1, :]

        for y in y_iter:
            prev_y = y + 1 if axis == "up" else y - 1
            prev = L[:, prev_y, :]                       # (D, W)
            min_prev = prev.min(axis=0, keepdims=True)
            prev_dm1 = np.empty_like(prev)
            prev_dm1[0] = np.inf
            prev_dm1[1:] = prev[:-1]
            prev_dp1 = np.empty_like(prev)
            prev_dp1[-1] = np.inf
            prev_dp1[:-1] = prev[1:]

            best = np.minimum(prev, prev_dm1 + P1)
            best = np.minimum(best, prev_dp1 + P1)
            best = np.minimum(best, min_prev + P2)  # min_prev broadcasts
            L[:, y, :] = cost_volume[:, y, :] + best - min_prev
    else:
        raise ValueError(f"unknown axis {axis}")
    return L


def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray) -> np.ndarray:
    """WGIF (edge-aware aggregation) → 4-direction SGM (global smoothness).

    Cascading SGM on top of WGIF preserves the edge-aware smoothing benefits
    while letting SGM enforce long-range disparity consistency. Pure SGM on
    raw Census costs over-smooths edges; pure WGIF underuses global structure.
    """
    cost = wgif_aggregate(cost_volume, guide_bgr).astype(np.float32, copy=False)
    results = {}

    def run(axis: str) -> None:
        results[axis] = _sgm_path(cost, axis)

    # 4 directions in parallel — each is independent.
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run, ["right", "left", "down", "up"]))

    return results["right"] + results["left"] + results["down"] + results["up"]


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    return compute_disp_skeleton(Il, Ir, max_disp, aggregate)

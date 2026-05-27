"""v7_iter — iterative refinement via confidence-aware 2-pass WGIF.

Pass 1 runs the standard WGIF aggregation. We then compute a per-pixel
confidence from the cost margin (how distinct is the winning disparity?) and
blend with a more aggressively smoothed second pass at low-confidence pixels.
Edges (where pass 1 is already confident) keep their sharpness; ambiguous
areas borrow from the smoother pass for cleaner cost surfaces.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .common import _MAX_WORKERS, compute_disp_skeleton


def _wgif(cost_volume: np.ndarray, guide_gray: np.ndarray,
          mean_I: np.ndarray, var_I: np.ndarray,
          ks: tuple, eps_base: float, eps0: float = 1e-6) -> np.ndarray:
    """Run WGIF with given guide stats. Returns filtered cost volume."""
    gamma = (var_I + eps0) / (var_I.mean() + eps0)
    denom = var_I + eps_base / gamma
    out = np.empty_like(cost_volume)

    def filter_slice(d: int) -> None:
        p = cost_volume[d]
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ks)
        mean_Ip = cv2.boxFilter(guide_gray * p, cv2.CV_32F, ks)
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / denom
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ks)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ks)
        out[d] = mean_a * guide_gray + mean_b

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        list(ex.map(filter_slice, range(cost_volume.shape[0])))
    return out


def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray,
              radius: int = 5, eps_sharp: float = 5e-4,
              eps_smooth: float = 5e-3) -> np.ndarray:
    """2-pass WGIF: sharp pass + smooth pass, blended by per-pixel confidence."""
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ks = (2 * radius + 1, 2 * radius + 1)

    # Shared guide stats.
    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ks)
    mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ks)
    var_I = mean_II - mean_I * mean_I

    # Pass 1 — sharp (small eps).
    cost_sharp = _wgif(cost_volume, guide_gray, mean_I, var_I, ks, eps_sharp)

    # Confidence from cost margin: well-defined winner = high confidence.
    sorted_cost = np.partition(cost_sharp, 1, axis=0)[:2]      # (2, H, W)
    margin = sorted_cost[1] - sorted_cost[0]
    # Normalise to [0, 1] roughly. Census max is 62, margins are smaller after filter.
    confidence = np.clip(margin / 5.0, 0.0, 1.0).astype(np.float32)

    # Pass 2 — strongly smoothed (large eps).
    cost_smooth = _wgif(cost_volume, guide_gray, mean_I, var_I, ks, eps_smooth)

    # Per-pixel blend: trust sharp pass where confident, smooth pass otherwise.
    alpha = confidence[None, :, :]  # broadcast (1, H, W)
    return alpha * cost_sharp + (1.0 - alpha) * cost_smooth


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    return compute_disp_skeleton(Il, Ir, max_disp, aggregate)

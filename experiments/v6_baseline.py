"""v6 baseline — Weighted Guided Image Filter aggregation (current production).

Per-pixel adaptive eps from local guide variance:

    γ(x)   = (var_I(x) + eps0) / mean(var_I + eps0)
    eps(x) = eps_base / γ(x)

γ is large at edges (small eps → preserve), small in textureless regions
(large eps → denoise). Content-driven, no max_disp/image-name branching.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .common import _MAX_WORKERS, compute_disp_skeleton


def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray,
              radius: int = 5, eps: float = 5e-4) -> np.ndarray:
    """WGIF aggregation (grayscale guide, per-pixel adaptive eps)."""
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ks = (2 * radius + 1, 2 * radius + 1)

    # Guide stats — once, shared across all disparity slices.
    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ks)
    mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ks)
    var_I = mean_II - mean_I * mean_I

    eps0 = 1e-6
    gamma = (var_I + eps0) / (var_I.mean() + eps0)
    denom = var_I + eps / gamma

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


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    return compute_disp_skeleton(Il, Ir, max_disp, aggregate)

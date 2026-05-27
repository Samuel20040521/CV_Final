"""v7_crossscale — Cross-scale cost aggregation (Zhang 2014, CVPR).

Build a 2-level pyramid of the cost volume + guide, run WGIF at each scale,
then upsample the coarse-scale result and blend per-pixel with the fine-scale
result. The coarse pass captures long-range smoothness; the fine pass keeps
edge sharpness.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .common import _MAX_WORKERS, compute_disp_skeleton


def _wgif_pass(cost_volume: np.ndarray, guide_gray: np.ndarray,
               radius: int, eps_base: float, eps0: float = 1e-6) -> np.ndarray:
    """Single WGIF pass."""
    ks = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ks)
    mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ks)
    var_I = mean_II - mean_I * mean_I
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
              radius: int = 5, eps: float = 5e-4,
              blend_alpha: float = 0.7) -> np.ndarray:
    """2-level pyramid WGIF.

    Fine pass at original resolution preserves edges.
    Coarse pass at 1/2 resolution captures broader smoothness.
    Final = blend_alpha * fine + (1-blend_alpha) * coarse_upsampled.
    """
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    D, H, W = cost_volume.shape

    # Fine pass at original resolution.
    cost_fine = _wgif_pass(cost_volume, guide_gray, radius=radius, eps_base=eps)

    # Build half-resolution cost and guide.
    H2, W2 = H // 2, W // 2
    guide_half = cv2.resize(guide_gray, (W2, H2), interpolation=cv2.INTER_LINEAR)
    # Downsample cost per disparity slice (INTER_AREA = box-averaging downsample).
    # WTA cares only about relative ordering, so no /4 scaling is needed.
    cost_half = np.empty((D, H2, W2), dtype=np.float32)
    for d in range(D):
        cost_half[d] = cv2.resize(cost_volume[d], (W2, H2), interpolation=cv2.INTER_AREA)

    cost_half_filtered = _wgif_pass(cost_half, guide_half, radius=radius, eps_base=eps)

    # Upsample coarse result back to original.
    cost_coarse = np.empty_like(cost_volume)
    for d in range(D):
        cost_coarse[d] = cv2.resize(cost_half_filtered[d], (W, H), interpolation=cv2.INTER_LINEAR)

    return blend_alpha * cost_fine + (1.0 - blend_alpha) * cost_coarse


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    return compute_disp_skeleton(Il, Ir, max_disp, aggregate)

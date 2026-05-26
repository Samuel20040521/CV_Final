"""NTU CV Spring 2026 Final Project — Stereo Matching.

Single-file disparity estimator implementing the standard Middlebury 4-step
pipeline: Census cost -> guided-filter aggregation -> winner-take-all ->
LR consistency + hole fill + weighted median.

Allowed cv2.ximgproc primitives used: createGuidedFilter, weightedMedianFilter.
Does NOT use cv2.StereoBM / cv2.StereoSGBM or any other built-in stereo solver.
Pure Python + numpy + opencv-contrib-python; no C extensions.

Public entry point referenced by eval.py and main.py:

    computeDisp(Il, Ir, max_disp) -> np.uint8 (H, W) disparity map
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# =============================================================================
# Step 1: Cost computation — Census transform + Hamming distance
# =============================================================================

CENSUS_WINDOW = (9, 7)  # (height, width) — 63 bits, fits in uint64


def _popcount_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = bin(i).count("1")
    return lut


_POPCOUNT8 = _popcount_lut()


def census_transform(gray: np.ndarray, window: Tuple[int, int] = CENSUS_WINDOW) -> np.ndarray:
    """Each pixel becomes a bitstring of (neighbor < center) comparisons.

    Returns uint64 array, shape (H, W). Border pixels use replicated padding so
    that the transform is defined everywhere (matching the OOB tip in the doc).
    """
    wh, ww = window
    assert wh * ww - 1 <= 64, "census window too large for uint64"
    h, w = gray.shape
    pad_h, pad_w = wh // 2, ww // 2
    padded = np.pad(gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge").astype(np.int16)
    center = padded[pad_h:pad_h + h, pad_w:pad_w + w]

    code = np.zeros((h, w), dtype=np.uint64)
    bit = 0
    for dy in range(wh):
        for dx in range(ww):
            if dy == pad_h and dx == pad_w:
                continue
            neighbor = padded[dy:dy + h, dx:dx + w]
            mask = (neighbor < center).astype(np.uint64)
            code |= mask << np.uint64(bit)
            bit += 1
    return code


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance of two uint64 arrays of equal shape."""
    x = np.bitwise_xor(a, b)
    bytes_view = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT8[bytes_view].sum(axis=-1).astype(np.float32)


def hamming_cost_volume(
    census_L: np.ndarray, census_R: np.ndarray, max_disp: int, direction: str = "L"
) -> np.ndarray:
    """Build a cost volume of shape (max_disp+1, H, W).

    direction='L': cost[d, y, x] = Hamming(L[y, x], R[y, x-d])  — disparity for the left image
    direction='R': cost[d, y, x] = Hamming(R[y, x], L[y, x+d])  — disparity for the right image

    Out-of-bound positions reuse the cost of the closest valid pixel (column clamp).
    """
    assert direction in ("L", "R")
    h, w = census_L.shape
    cost = np.empty((max_disp + 1, h, w), dtype=np.float32)
    x_idx = np.arange(w)
    for d in range(max_disp + 1):
        if direction == "L":
            shifted_idx = np.clip(x_idx - d, 0, w - 1)
            cost[d] = _hamming(census_L, census_R[:, shifted_idx])
        else:
            shifted_idx = np.clip(x_idx + d, 0, w - 1)
            cost[d] = _hamming(census_R, census_L[:, shifted_idx])
    return cost


# =============================================================================
# Step 2: Cost aggregation — guided filter per disparity slice
# =============================================================================

def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray, radius: int = 7, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware aggregation: apply a guided filter to each disparity slice.

    Guided filter is O(1) in radius and ~10x faster than the joint bilateral
    filter at this volume size, with comparable quality on Middlebury.
    """
    guide = guide_bgr.astype(np.float32) / 255.0
    gf = cv2.ximgproc.createGuidedFilter(guide=guide, radius=radius, eps=eps)
    out = np.empty_like(cost_volume)
    for d in range(cost_volume.shape[0]):
        out[d] = gf.filter(cost_volume[d])
    return out


# =============================================================================
# Step 3: Disparity optimization — winner-take-all
# =============================================================================

def winner_take_all(cost_volume: np.ndarray) -> np.ndarray:
    """Per-pixel argmin disparity as int32, shape (H, W)."""
    return np.argmin(cost_volume, axis=0).astype(np.int32)


# =============================================================================
# Step 4: Refinement — LR consistency + hole fill + weighted median
# =============================================================================

def lr_consistency(D_L: np.ndarray, D_R: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Boolean validity mask. True where the L disparity is consistent with R."""
    h, w = D_L.shape
    x_idx = np.broadcast_to(np.arange(w), (h, w))
    corr_x = x_idx - D_L
    in_bounds = corr_x >= 0
    safe_x = np.where(in_bounds, corr_x, 0)
    y_idx = np.broadcast_to(np.arange(h)[:, None], (h, w))
    matched_R = D_R[y_idx, safe_x]
    diff = np.abs(D_L - matched_R)
    return in_bounds & (diff <= threshold)


def hole_fill(D_L: np.ndarray, valid: np.ndarray, max_disp: int) -> np.ndarray:
    """Fill invalid pixels with min(nearest-valid-left, nearest-valid-right) per scanline.

    Per the doc's boundary tip, leading/trailing invalid runs are padded with max_disp
    so an entire invalid scanline edge does not collapse to 0.
    """
    h, w = D_L.shape
    D_fill_L = D_L.copy()
    D_fill_R = D_L.copy()
    last = np.full(h, max_disp, dtype=D_L.dtype)
    for x in range(w):
        col_valid = valid[:, x]
        last = np.where(col_valid, D_L[:, x], last)
        D_fill_L[:, x] = last
    last = np.full(h, max_disp, dtype=D_L.dtype)
    for x in range(w - 1, -1, -1):
        col_valid = valid[:, x]
        last = np.where(col_valid, D_L[:, x], last)
        D_fill_R[:, x] = last
    return np.minimum(D_fill_L, D_fill_R)


def weighted_median(disp: np.ndarray, guide_bgr: np.ndarray, radius: int = 15) -> np.ndarray:
    """Edge-aware weighted median to clean up the disparity map."""
    disp_u8 = disp.astype(np.uint8)
    out = cv2.ximgproc.weightedMedianFilter(joint=guide_bgr, src=disp_u8, r=radius)
    return out.astype(np.int32)


def refine(D_L: np.ndarray, D_R: np.ndarray, guide_bgr: np.ndarray, max_disp: int) -> np.ndarray:
    valid = lr_consistency(D_L, D_R)
    filled = hole_fill(D_L, valid, max_disp)
    return weighted_median(filled, guide_bgr)


# =============================================================================
# Public entry point
# =============================================================================

def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    """Stereo disparity estimation for the left image.

    Args:
        Il, Ir: BGR uint8 images from cv2.imread, shape (H, W, 3).
        max_disp: integer maximum disparity to search.

    Returns:
        Disparity map for the left image, uint8 shape (H, W), values in [0, max_disp].
    """
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    # Step 1: cost volumes (both directions, for the later LR consistency check)
    census_L = census_transform(gray_L)
    census_R = census_transform(gray_R)
    cost_L = hamming_cost_volume(census_L, census_R, max_disp, direction="L")
    cost_R = hamming_cost_volume(census_L, census_R, max_disp, direction="R")

    # Step 2: edge-aware aggregation
    cost_L = aggregate(cost_L, Il)
    cost_R = aggregate(cost_R, Ir)

    # Step 3: winner-take-all
    D_L = winner_take_all(cost_L)
    D_R = winner_take_all(cost_R)

    # Step 4: LRC + hole fill + weighted median
    labels = refine(D_L, D_R, Il, max_disp)

    return labels.astype(np.uint8)


__all__ = ["computeDisp"]

"""Shared building blocks for stereo-matching variants.

Each variant lives in `experiments/v?_xxx.py` and exposes a `computeDisp(Il, Ir,
max_disp)` function. Most of the pipeline (Census transform, Hamming cost,
reindex, WTA, LRC, hole-fill, sub-pixel, weighted-median refinement) is shared
and lives here; what each variant *differs* in is the `aggregate(cost_volume,
guide_bgr) -> np.ndarray` callable they pass to `compute_disp_skeleton`.

Hyperparameters baked in here are constant across all images — no branching on
max_disp or image name, in compliance with the TA's 2026-05-27 rule.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple

import cv2
import numpy as np


# =============================================================================
# Constants & runtime feature detection
# =============================================================================

CENSUS_WINDOW = (9, 7)  # 63 bits, fits in uint64
_MAX_WORKERS = min(os.cpu_count() or 4, 8)
_HAS_NP_BITWISE_COUNT = hasattr(np, "bitwise_count")

# SWAR popcount mask constants (used when numpy.bitwise_count is unavailable).
_SWAR_M1 = np.uint64(0x5555555555555555)
_SWAR_M2 = np.uint64(0x3333333333333333)
_SWAR_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SWAR_H  = np.uint64(0x0101010101010101)
_SWAR_1  = np.uint64(1)
_SWAR_2  = np.uint64(2)
_SWAR_4  = np.uint64(4)
_SWAR_56 = np.uint64(56)


def _popcount_swar(x: np.ndarray) -> np.ndarray:
    """Vectorised SWAR popcount over a uint64 array."""
    x = x - ((x >> _SWAR_1) & _SWAR_M1)
    x = (x & _SWAR_M2) + ((x >> _SWAR_2) & _SWAR_M2)
    x = (x + (x >> _SWAR_4)) & _SWAR_M4
    return (x * _SWAR_H) >> _SWAR_56


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance of two uint64 arrays of equal shape."""
    x = np.bitwise_xor(a, b)
    if _HAS_NP_BITWISE_COUNT:
        return np.bitwise_count(x).astype(np.float32)
    return _popcount_swar(x).astype(np.float32)


# =============================================================================
# Step 1: Cost computation
# =============================================================================

def census_transform(gray: np.ndarray, window: Tuple[int, int] = CENSUS_WINDOW) -> np.ndarray:
    """Each pixel becomes a bitstring of (neighbor < center) comparisons."""
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


def hamming_cost_volume(
    census_L: np.ndarray, census_R: np.ndarray, max_disp: int
) -> np.ndarray:
    """Left-anchored cost volume of shape (max_disp+1, H, W), per-disparity threaded."""
    h, w = census_L.shape
    cost = np.empty((max_disp + 1, h, w), dtype=np.float32)
    x_idx = np.arange(w)

    def compute_slice(d: int) -> None:
        shifted_idx = np.clip(x_idx - d, 0, w - 1)
        cost[d] = _hamming(census_L, census_R[:, shifted_idx])

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        list(ex.map(compute_slice, range(max_disp + 1)))
    return cost


def reindex_cost_to_right(cost_L: np.ndarray) -> np.ndarray:
    """Derive R-direction cost volume from cost_L via column reindex."""
    D, h, w = cost_L.shape
    x_idx = np.arange(w)[None, :]
    d_idx = np.arange(D)[:, None]
    src_idx_2d = np.clip(x_idx + d_idx, 0, w - 1)
    src_idx_3d = np.broadcast_to(src_idx_2d[:, None, :], cost_L.shape)
    return np.take_along_axis(cost_L, src_idx_3d, axis=2)


# =============================================================================
# Step 3: Disparity optimization
# =============================================================================

def winner_take_all(cost_volume: np.ndarray) -> np.ndarray:
    return np.argmin(cost_volume, axis=0).astype(np.int32)


# =============================================================================
# Step 4: Refinement
# =============================================================================

def lr_consistency(D_L: np.ndarray, D_R: np.ndarray, threshold: int = 1) -> np.ndarray:
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


def subpixel_refine(D_int: np.ndarray, cost: np.ndarray, max_disp: int) -> np.ndarray:
    """Parabolic sub-pixel refinement around the WTA argmin."""
    D = D_int.astype(np.int32)
    H, W = D.shape
    y_idx = np.arange(H)[:, None]
    x_idx = np.arange(W)[None, :]
    D_safe = np.clip(D, 1, max_disp - 1)
    c_m = cost[D_safe - 1, y_idx, x_idx]
    c_0 = cost[D_safe,     y_idx, x_idx]
    c_p = cost[D_safe + 1, y_idx, x_idx]
    denom = c_m + c_p - 2.0 * c_0
    nonflat = np.abs(denom) > 1e-6
    delta = np.where(nonflat, (c_m - c_p) / (2.0 * np.where(nonflat, denom, 1.0)), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    d_sub = D.astype(np.float32) + delta
    boundary = (D == 0) | (D == max_disp)
    return np.where(boundary, D.astype(np.float32), d_sub)


def weighted_median_subpixel(
    disp_float: np.ndarray, guide_bgr: np.ndarray, max_disp: int, radius: int = 15
) -> np.ndarray:
    """WMF on the sub-pixel disparity, cast directly to uint8."""
    disp_u8 = np.clip(disp_float, 0.0, float(max_disp)).astype(np.uint8)
    filtered = cv2.ximgproc.weightedMedianFilter(joint=guide_bgr, src=disp_u8, r=radius)
    return filtered.astype(np.int32)


def refine(
    D_L: np.ndarray, D_R: np.ndarray, cost_L: np.ndarray,
    guide_bgr: np.ndarray, max_disp: int,
) -> np.ndarray:
    """LRC + hole fill + sub-pixel + weighted median."""
    valid = lr_consistency(D_L, D_R)
    filled_int = hole_fill(D_L, valid, max_disp).astype(np.float32)
    d_sub = subpixel_refine(D_L, cost_L, max_disp)
    d_combined = np.where(valid, d_sub, filled_int)
    return weighted_median_subpixel(d_combined, guide_bgr, max_disp)


# =============================================================================
# Standard pipeline skeleton — each variant plugs in its own aggregate_fn.
# =============================================================================

def compute_disp_skeleton(
    Il: np.ndarray, Ir: np.ndarray, max_disp: int,
    aggregate_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Standard 4-step pipeline; only Step 2 (aggregation) is variant-specific.

    Args:
        Il, Ir: BGR uint8 stereo pair.
        max_disp: search range.
        aggregate_fn: function (cost_volume, guide_bgr) -> filtered cost_volume.

    Returns:
        uint8 disparity map (H, W).
    """
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    # Step 1: Census + Hamming, R via reindex.
    census_L = census_transform(gray_L)
    census_R = census_transform(gray_R)
    cost_L = hamming_cost_volume(census_L, census_R, max_disp)
    cost_R_raw = reindex_cost_to_right(cost_L)

    # Step 2: variant-specific aggregation on L cost only.
    cost_L_agg = aggregate_fn(cost_L, Il)

    # Step 3: winner-take-all (L from aggregated, R from raw — saves a filter pass).
    D_L = winner_take_all(cost_L_agg)
    D_R = winner_take_all(cost_R_raw)

    # Step 4: refinement.
    labels = refine(D_L, D_R, cost_L_agg, Il, max_disp)
    return labels.astype(np.uint8)

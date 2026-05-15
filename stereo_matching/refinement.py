"""Step 4: Refinement — LR consistency check, hole filling, weighted median filter."""
from __future__ import annotations

import cv2
import numpy as np


def lr_consistency(D_L: np.ndarray, D_R: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Boolean validity mask: True where the L disparity is consistent with R.

    Pixel (y, x) is valid iff |D_L[y, x] - D_R[y, x - D_L[y, x]]| <= threshold.
    Pixels whose corresponding right-image position falls out of bounds are invalid.
    """
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
    """Fill invalid pixels per scanline with min(nearest-valid-left, nearest-valid-right).

    Per the doc's boundary tip, leading/trailing invalid runs are padded with max_disp
    so that an entire invalid scanline edge does not collapse to 0.
    """
    h, w = D_L.shape
    D_fill_L = D_L.copy()
    D_fill_R = D_L.copy()
    # Sweep left → right: carry forward the most recent valid disparity; seed with max_disp.
    last = np.full(h, max_disp, dtype=D_L.dtype)
    for x in range(w):
        col_valid = valid[:, x]
        last = np.where(col_valid, D_L[:, x], last)
        D_fill_L[:, x] = last
    # Sweep right → left.
    last = np.full(h, max_disp, dtype=D_L.dtype)
    for x in range(w - 1, -1, -1):
        col_valid = valid[:, x]
        last = np.where(col_valid, D_L[:, x], last)
        D_fill_R[:, x] = last
    return np.minimum(D_fill_L, D_fill_R)


def weighted_median(disp: np.ndarray, guide_bgr: np.ndarray, radius: int = 17) -> np.ndarray:
    """Edge-aware weighted median filter to clean up the disparity map."""
    disp_u8 = disp.astype(np.uint8)
    out = cv2.ximgproc.weightedMedianFilter(joint=guide_bgr, src=disp_u8, r=radius)
    return out.astype(np.int32)


def refine(
    D_L: np.ndarray, D_R: np.ndarray, guide_bgr: np.ndarray, max_disp: int
) -> np.ndarray:
    valid = lr_consistency(D_L, D_R)
    filled = hole_fill(D_L, valid, max_disp)
    return weighted_median(filled, guide_bgr)

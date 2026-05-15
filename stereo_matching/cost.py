"""Step 1: Cost computation via Census transform + Hamming distance."""
from __future__ import annotations

import numpy as np

CENSUS_WINDOW = (9, 7)  # (height, width) — 63 bits, fits in uint64


def _popcount_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = bin(i).count("1")
    return lut


_POPCOUNT8 = _popcount_lut()


def census_transform(gray: np.ndarray, window: tuple[int, int] = CENSUS_WINDOW) -> np.ndarray:
    """Census transform: each pixel becomes a bitstring of (neighbor < center) comparisons.

    Returns uint64 array, shape (H, W). Border pixels use replicated padding so that the
    transform is defined everywhere (matching the OOB tip from the doc).
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
    # Split uint64 into 8 bytes and sum popcounts via LUT.
    bytes_view = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT8[bytes_view].sum(axis=-1).astype(np.float32)


def hamming_cost_volume(
    census_L: np.ndarray, census_R: np.ndarray, max_disp: int, direction: str = "L"
) -> np.ndarray:
    """Build a cost volume of shape (max_disp+1, H, W).

    direction='L': cost[d, y, x] = Hamming(L[y,x], R[y, x-d])  — disparity for the left image
    direction='R': cost[d, y, x] = Hamming(R[y,x], L[y, x+d])  — disparity for the right image

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

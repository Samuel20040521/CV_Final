"""Step 2: Edge-aware cost aggregation using a guided filter per disparity slice."""
from __future__ import annotations

import cv2
import numpy as np


def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray, radius: int = 11, eps: float = 1e-4) -> np.ndarray:
    """Apply a guided filter to each disparity slice of the cost volume.

    The guided filter is O(1) in radius and ~10× faster than the joint bilateral filter
    at this volume size, with comparable quality on Middlebury — important for staying
    within the 10-min budget when max_disp=60.
    """
    guide = guide_bgr.astype(np.float32) / 255.0
    gf = cv2.ximgproc.createGuidedFilter(guide=guide, radius=radius, eps=eps)
    out = np.empty_like(cost_volume)
    for d in range(cost_volume.shape[0]):
        out[d] = gf.filter(cost_volume[d])
    return out

"""computeDisp orchestrator — the public entry point referenced by eval.py and main.py."""
from __future__ import annotations

import cv2
import numpy as np

from .cost import census_transform, hamming_cost_volume
from .aggregation import aggregate
from .optimization import winner_take_all
from .refinement import refine


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    """Stereo disparity estimation, Middlebury 4-step pipeline.

    Args:
        Il, Ir: BGR uint8 images from cv2.imread, shape (H, W, 3).
        max_disp: integer maximum disparity to search.

    Returns:
        Disparity map for the left image, uint8 shape (H, W), values in [0, max_disp].
    """
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    # Step 1 — Cost computation (both directions, for the later LR consistency check).
    census_L = census_transform(gray_L)
    census_R = census_transform(gray_R)
    cost_L = hamming_cost_volume(census_L, census_R, max_disp, direction="L")
    cost_R = hamming_cost_volume(census_L, census_R, max_disp, direction="R")

    # Step 2 — Edge-aware aggregation, guided by the corresponding image.
    cost_L = aggregate(cost_L, Il)
    cost_R = aggregate(cost_R, Ir)

    # Step 3 — Winner-take-all.
    D_L = winner_take_all(cost_L)
    D_R = winner_take_all(cost_R)

    # Step 4 — LR consistency → hole fill → weighted median.
    labels = refine(D_L, D_R, Il, max_disp)

    return labels.astype(np.uint8)

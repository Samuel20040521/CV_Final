"""Step 3: Winner-take-all disparity selection."""
from __future__ import annotations

import numpy as np


def winner_take_all(cost_volume: np.ndarray) -> np.ndarray:
    """Return per-pixel argmin disparity as int32, shape (H, W)."""
    return np.argmin(cost_volume, axis=0).astype(np.int32)

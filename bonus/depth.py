"""Disparity → metric depth → colored point cloud.

After rectification the relation between disparity d (pixels) and depth Z
(metres) is exact:

    Z = f * B / d

where f is the rectified focal length (pixels) and B is the baseline (metres).
Pixels with disparity 0 (or below a small threshold) are invalid — typically
because they're at infinity, in a hole left by LR consistency, or beyond the
search range.
"""
from __future__ import annotations

import numpy as np


def disparity_to_depth(disp: np.ndarray, fx: float, baseline: float, min_disp: float = 1.0) -> np.ndarray:
    """Return depth map in metres; pixels with disp < min_disp become 0."""
    depth = np.zeros_like(disp, dtype=np.float32)
    mask = disp >= min_disp
    depth[mask] = (fx * baseline) / disp[mask].astype(np.float32)
    return depth


def depth_to_points(
    depth: np.ndarray,
    color_bgr: np.ndarray,
    K: np.ndarray,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map into a colored point cloud in the camera frame.

    Args:
        depth: (H, W) float32 metres, 0 means invalid.
        color_bgr: (H, W, 3) uint8, aligned with depth.
        K: (3, 3) rectified intrinsics.
        max_depth: drop points farther than this — KITTI scenes top out around
            80 m for usable LiDAR returns and stereo gets unreliable past that.

    Returns:
        (points, colors) as (N, 3) float32 and (N, 3) uint8 (RGB, Open3D style).
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    valid = (Z > 0) & (Z < max_depth)
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=-1).astype(np.float32)

    # BGR → RGB
    colors = color_bgr[valid][..., ::-1]
    return pts, colors.astype(np.uint8)

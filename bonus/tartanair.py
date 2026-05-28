"""TartanAir V1 dataset loader for the bonus 3D fusion pipeline.

Expected layout (produced by ``bonus.download_tartanair``)::

    data/tartanair_raw/<scene>/<level>/P00X/
    ├── image_left/  000XXX_left.png       640x480 PNG, rectified
    ├── image_right/ 000XXX_right.png      baseline 0.25 m
    ├── depth_left/  000XXX_left_depth.npy (480, 640) float32 metres
    ├── pose_left.txt                      one row per frame: tx ty tz qx qy qz qw
    └── pose_right.txt                     (not used by this loader)

Coordinate / pose convention used by TartanAir V1: each row of
``pose_left.txt`` is the *world-frame position and orientation* of the left
camera, expressed using **NED body axes** (x=forward, y=right, z=down).
The orientation is a unit quaternion (Hamilton convention) ordered
``(qx, qy, qz, qw)``.

This loader returns ``T_world_cam`` already converted into the **OpenCV
camera convention** (x=right, y=down, z=forward) that ``depth_left.npy``,
``cv2.imread`` images, and Open3D's TSDF integrator all use. The conversion
is a right-multiplication by a fixed axis permutation matrix ``M_NED2CV``::

    OpenCV x (right)   = NED y (right)
    OpenCV y (down)    = NED z (down)
    OpenCV z (forward) = NED x (forward)

i.e. ``T_world_cam_cv = T_world_cam_ned @ M_NED2CV``. Without this
correction, every per-frame depth gets back-projected with z=forward but
then transformed by a pose whose forward axis is x, so all reconstructed
points land along world-z (vertical) instead of along the alley.

Empirical sanity check (verified during dataset setup): on P000 frame 0,
``|t_left - t_right| = 0.2500 m`` — matches the documented baseline exactly,
confirming that the stored translations are in the same units (metres) and
the rotation puts left/right cams in the same world.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


# Intrinsics are constant across the entire TartanAir V1 dataset.
FX = 320.0
FY = 320.0
CX = 320.0
CY = 240.0
BASELINE_M = 0.25
IMAGE_HW = (480, 640)

# TartanAir marks sky / past-far-plane pixels with depth ≈ float16-max (65504).
# Anything above this threshold should be treated as invalid before fusion.
SKY_DEPTH_THRESHOLD = 65000.0


@dataclass
class Frame:
    """File paths for one TartanAir frame.

    ``image_left`` / ``image_right`` are 640×480 BGR (after ``cv2.imread``);
    ``depth_left`` is a ``(480, 640) float32`` array in metres after ``np.load``.
    """
    image_left: Path
    image_right: Path
    depth_left: Path


def get_intrinsics() -> np.ndarray:
    """Return the (3, 3) intrinsics matrix shared by the left and right cams."""
    return np.array([
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def list_frames(traj_dir: Path, indices) -> List[Frame]:
    """Build :class:`Frame` records for the requested indices.

    Args:
        traj_dir: e.g. ``data/tartanair_raw/japanesealley/Hard/P000``.
        indices: any iterable of int (typically ``range(start, end, stride)``).

    Raises:
        FileNotFoundError if any of the expected files is missing for one of
        the requested indices.
    """
    out: List[Frame] = []
    for i in indices:
        stem = "{:06d}".format(i)
        L = traj_dir / "image_left" / "{}_left.png".format(stem)
        R = traj_dir / "image_right" / "{}_right.png".format(stem)
        D = traj_dir / "depth_left" / "{}_left_depth.npy".format(stem)
        if not (L.is_file() and R.is_file() and D.is_file()):
            raise FileNotFoundError(
                "Missing files for frame {} under {} (L={}, R={}, D={})".format(
                    i, traj_dir, L.exists(), R.exists(), D.exists()))
        out.append(Frame(L, R, D))
    return out


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (Hamilton, x-y-z-w order) → 3×3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def load_poses(traj_dir: Path, indices) -> List[np.ndarray]:
    """Return ``T_world_cam`` (4×4) for each requested frame index.

    The matrix is already in **OpenCV camera convention** (see module
    docstring) — feed directly into TSDF integration / rendering without
    further axis flips.

    Reads the full ``pose_left.txt`` once and slices the requested rows.
    """
    rows = np.loadtxt(traj_dir / "pose_left.txt")
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise RuntimeError("Unexpected pose_left.txt shape {} (expected (N, 7))".format(rows.shape))

    # NED body axes (x=fwd, y=right, z=down) → OpenCV cam axes (x=right, y=down,
    # z=fwd). Right-multiply each pose by this 4×4 permutation.
    M_NED2CV = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    out: List[np.ndarray] = []
    for i in indices:
        row = rows[i]
        t = row[:3]
        qx, qy, qz, qw = row[3], row[4], row[5], row[6]
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = t
        out.append(T @ M_NED2CV)
    return out

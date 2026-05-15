"""KITTI raw dataset loader.

Expected layout (the standard KITTI raw distribution):

    <kitti_root>/
    └── 2011_09_26/
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        ├── calib_velo_to_cam.txt
        └── 2011_09_26_drive_0005_sync/
            ├── image_02/data/0000000000.png ...     (rectified color left)
            ├── image_03/data/0000000000.png ...     (rectified color right)
            └── oxts/data/0000000000.txt ...         (GPS/IMU per frame)

We parse the unrectified intrinsics + the rectification rotation that KITTI
publishes (K_unrect_xx, R_rect_xx, P_rect_xx) so that bonus/rectify.py can
reconstruct rectification homographies from first principles rather than
relying on the already-rectified image_02/03 streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration parsing                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class CamCalib:
    """All quantities needed to rectify a stereo pair from scratch.

    Subscripts 0/1 follow KITTI: cam 2 (color left) is index 2, cam 3 (color
    right) is index 3 in the raw calibration file; we re-label them L/R here.
    """
    K_L_unrect: np.ndarray   # (3, 3) intrinsics of left color cam pre-rect
    K_R_unrect: np.ndarray   # (3, 3) intrinsics of right color cam pre-rect
    D_L: np.ndarray          # (5,)   left distortion
    D_R: np.ndarray          # (5,)   right distortion
    S_L: tuple[int, int]     # (width, height) of left image
    S_R: tuple[int, int]
    R_L: np.ndarray          # (3, 3) rotation from left cam to its rect frame
    R_R: np.ndarray
    P_L_rect: np.ndarray     # (3, 4) projection matrix into rect left frame
    P_R_rect: np.ndarray     # (3, 4) projection matrix into rect right frame
    T_LR: np.ndarray         # (3,)   baseline translation between left/right
                             #        rectified cameras, in metres


def _parse_calib_file(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        nums = [float(x) for x in rest.split()]
        if not nums:
            continue
        out[key] = np.asarray(nums, dtype=np.float64)
    return out


def load_calibration(date_dir: Path) -> CamCalib:
    """Read calib_cam_to_cam.txt for the given KITTI date directory."""
    raw = _parse_calib_file(date_dir / "calib_cam_to_cam.txt")

    def K(idx: str) -> np.ndarray:
        return raw[f"K_{idx}"].reshape(3, 3)

    def S(idx: str) -> tuple[int, int]:
        w, h = raw[f"S_{idx}"]
        return int(w), int(h)

    def Rrect(idx: str) -> np.ndarray:
        return raw[f"R_rect_{idx}"].reshape(3, 3)

    def Prect(idx: str) -> np.ndarray:
        return raw[f"P_rect_{idx}"].reshape(3, 4)

    # Translation between the two rectified cameras: from P_rect_03 the
    # principal point shift gives baseline directly: T_x = -P[0, 3] / P[0, 0].
    P2 = Prect("02")
    P3 = Prect("03")
    fx = P2[0, 0]
    baseline = (P2[0, 3] - P3[0, 3]) / fx  # metres, positive number

    return CamCalib(
        K_L_unrect=K("02"),
        K_R_unrect=K("03"),
        D_L=raw["D_02"],
        D_R=raw["D_03"],
        S_L=S("02"),
        S_R=S("03"),
        R_L=Rrect("02"),
        R_R=Rrect("03"),
        P_L_rect=P2,
        P_R_rect=P3,
        T_LR=np.array([baseline, 0.0, 0.0], dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# OXTS (GPS/IMU) → SE(3) poses                                                 #
# --------------------------------------------------------------------------- #

# Field order from KITTI's oxts/dataformat.txt (the first 6 columns are what we
# care about: lat lon alt roll pitch yaw).

def _oxts_to_pose(oxts: np.ndarray, scale: float, origin: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OXTS record to a 4x4 IMU-to-world pose using Mercator projection.

    Returns (T_world_imu, origin_used). origin_used should be passed back as
    `origin` for subsequent frames so the first frame anchors world coords.
    """
    lat, lon, alt = oxts[0], oxts[1], oxts[2]
    roll, pitch, yaw = oxts[3], oxts[4], oxts[5]

    er = 6378137.0
    mx = scale * lon * np.pi * er / 180.0
    my = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    t = np.array([mx, my, alt], dtype=np.float64)
    if origin is None:
        origin = t.copy()
    t = t - origin

    # ZYX intrinsic = yaw * pitch * roll (per KITTI's devkit)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, origin


def _mercator_scale(lat0: float) -> float:
    return np.cos(lat0 * np.pi / 180.0)


def load_poses(drive_dir: Path, indices: range) -> list[np.ndarray]:
    """Return world←IMU poses (4x4) for the requested frame indices.

    Pose 0's translation defines the world origin so the trajectory starts at
    (0, 0, alt_0). Rotation is left absolute — useful for fusion.
    """
    oxts_dir = drive_dir / "oxts" / "data"
    files = sorted(oxts_dir.glob("*.txt"))
    first = np.fromstring(files[indices.start].read_text().split("\n")[0], sep=" ")
    scale = _mercator_scale(first[0])
    origin: np.ndarray | None = None
    poses: list[np.ndarray] = []
    for i in indices:
        rec = np.fromstring(files[i].read_text().split("\n")[0], sep=" ")
        T, origin = _oxts_to_pose(rec, scale, origin)
        poses.append(T)
    return poses


def load_imu_to_cam(date_dir: Path) -> np.ndarray:
    """Return the 4x4 transform from IMU frame to the *rectified left* camera frame.

    KITTI ships imu→velo and velo→cam0; we compose them and then apply R_rect_00
    to land in cam0's rectified frame, which coincides with the rectified left
    color cam's frame after rectification translation.
    """
    imu_to_velo = _parse_calib_file(date_dir / "calib_imu_to_velo.txt")
    velo_to_cam = _parse_calib_file(date_dir / "calib_velo_to_cam.txt")

    R_iv = imu_to_velo["R"].reshape(3, 3)
    t_iv = imu_to_velo["T"].reshape(3)
    R_vc = velo_to_cam["R"].reshape(3, 3)
    t_vc = velo_to_cam["T"].reshape(3)

    T_iv = np.eye(4); T_iv[:3, :3] = R_iv; T_iv[:3, 3] = t_iv
    T_vc = np.eye(4); T_vc[:3, :3] = R_vc; T_vc[:3, 3] = t_vc

    # Rectification rotation lives in cam_to_cam under R_rect_00.
    cc = _parse_calib_file(date_dir / "calib_cam_to_cam.txt")
    R_rect_00 = cc["R_rect_00"].reshape(3, 3)
    T_rect = np.eye(4); T_rect[:3, :3] = R_rect_00

    return T_rect @ T_vc @ T_iv


# --------------------------------------------------------------------------- #
# Image iteration                                                              #
# --------------------------------------------------------------------------- #

def list_pairs(drive_dir: Path) -> list[tuple[Path, Path]]:
    """Sorted list of (left_color_path, right_color_path)."""
    left = sorted((drive_dir / "image_02" / "data").glob("*.png"))
    right = sorted((drive_dir / "image_03" / "data").glob("*.png"))
    if len(left) != len(right):
        raise RuntimeError(f"left/right frame counts differ: {len(left)} vs {len(right)}")
    return list(zip(left, right))

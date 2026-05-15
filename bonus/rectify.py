"""Calibrated stereo rectification, implemented from first principles.

Given a calibrated stereo pair (intrinsics K_L, K_R, rotation R between them and
baseline T), the textbook rectification finds two rotations R_rect_L, R_rect_R
so that, after applying them, the two virtual cameras share an image plane and
their x-axes lie along the baseline. The remaining mapping from raw pixel to
rectified pixel is then a homography:

    H = K_new · R_rect · K_orig⁻¹

We construct (R_rect_L, R_rect_R) directly from R and T rather than calling
`cv2.stereoRectify`. We then build the per-pixel remap lookup tables once and
use `cv2.remap` for the actual warp (this is just bilinear interpolation, so
it's outside the "develop and implement" scope and would be tedious to
re-implement).

The KITTI raw distribution already publishes R_rect_xx and P_rect_xx; we expose
a separate `rectification_from_kitti` helper so the bonus pipeline can use
those exactly, which guarantees we match the KITTI ground-truth rectification
and avoids drift from re-deriving them. Both paths produce the same kind of
remap LUTs.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class StereoRectification:
    """Everything needed to convert raw L/R images to rectified L/R images."""
    map_Lx: np.ndarray     # float32 (H, W) for cv2.remap
    map_Ly: np.ndarray
    map_Rx: np.ndarray
    map_Ry: np.ndarray
    K_rect_L: np.ndarray   # (3, 3) intrinsics of rectified left cam
    K_rect_R: np.ndarray
    baseline: float        # metres
    out_size: tuple[int, int]  # (W, H)


# --------------------------------------------------------------------------- #
# From-scratch rectification (textbook decomposition)                          #
# --------------------------------------------------------------------------- #

def _rectification_rotations(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (R_rect_L, R_rect_R) so that both cams share an image plane.

    Algorithm (Trucco & Verri, "Introductory Techniques for 3-D Computer Vision"):
      e1 = T / |T|                       (new x-axis = baseline direction)
      e2 = (-Ty, Tx, 0) / sqrt(Tx² + Ty²) (new y-axis ⟂ e1 and to old optical axis)
      e3 = e1 × e2                       (new z-axis completes the right-handed frame)
      R_rect_L = [e1; e2; e3]
      R_rect_R = R_rect_L · R
    """
    Tx, Ty, Tz = T
    e1 = T / np.linalg.norm(T)
    e2 = np.array([-Ty, Tx, 0.0])
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R_rect_L = np.stack([e1, e2, e3], axis=0)
    R_rect_R = R_rect_L @ R
    return R_rect_L, R_rect_R


def _rectification_homography(R_rect: np.ndarray, K_orig: np.ndarray, K_new: np.ndarray) -> np.ndarray:
    """H mapping raw pixel coordinates → rectified pixel coordinates.

    The inverse is what cv2.remap actually wants (sample-from coords), so we
    return H itself and let the caller invert.
    """
    return K_new @ R_rect @ np.linalg.inv(K_orig)


def build_rectification(
    K_L: np.ndarray,
    K_R: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: tuple[int, int],
    D_L: np.ndarray | None = None,
    D_R: np.ndarray | None = None,
) -> StereoRectification:
    """From-scratch rectification.

    Args:
        K_L, K_R: original intrinsics (3x3)
        R, T: rotation/translation of right cam expressed in the left cam frame
        image_size: (width, height) of raw images
        D_L, D_R: optional distortion vectors (5,); if provided we undistort as
            part of the remap LUT.
    """
    W, H = image_size
    R_rect_L, R_rect_R = _rectification_rotations(R, T)

    # Choose a shared new intrinsic K_new. Use the mean focal length of the two
    # cameras and an output principal point in the middle of the image.
    fx = 0.5 * (K_L[0, 0] + K_R[0, 0])
    fy = 0.5 * (K_L[1, 1] + K_R[1, 1])
    cx, cy = W / 2.0, H / 2.0
    K_new = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Build the per-pixel sample-from map: for each output pixel (u, v), find
    # the corresponding raw-image pixel. That's the *inverse* of the rectifying
    # transform: raw = (R_rect @ K_new⁻¹ · [u, v, 1])  (in camera ray space)
    # which projects back through K_orig (after optional distortion).
    map_Lx, map_Ly = _build_remap(K_new, R_rect_L, K_L, D_L, (W, H))
    map_Rx, map_Ry = _build_remap(K_new, R_rect_R, K_R, D_R, (W, H))

    baseline = float(np.linalg.norm(T))
    return StereoRectification(
        map_Lx=map_Lx, map_Ly=map_Ly, map_Rx=map_Rx, map_Ry=map_Ry,
        K_rect_L=K_new, K_rect_R=K_new, baseline=baseline, out_size=(W, H),
    )


def _build_remap(
    K_new: np.ndarray, R_rect: np.ndarray, K_orig: np.ndarray,
    D: np.ndarray | None, size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    W, H = size
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pts_rect = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)

    # rectified pixel → rectified normalized ray → original cam coords
    inv_K_new = np.linalg.inv(K_new)
    rays = pts_rect @ inv_K_new.T            # (H, W, 3) in rectified cam frame
    rays = rays @ R_rect                     # rotate back into original cam frame
    rays /= rays[..., 2:3]                   # normalize to z = 1

    if D is None:
        x_n, y_n = rays[..., 0], rays[..., 1]
    else:
        x_n, y_n = _apply_distortion(rays[..., 0], rays[..., 1], D)

    # Project through original intrinsics.
    fx, fy = K_orig[0, 0], K_orig[1, 1]
    cx, cy = K_orig[0, 2], K_orig[1, 2]
    map_x = (fx * x_n + cx).astype(np.float32)
    map_y = (fy * y_n + cy).astype(np.float32)
    return map_x, map_y


def _apply_distortion(x: np.ndarray, y: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brown's plumb-bob distortion (k1, k2, p1, p2, k3)."""
    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


# --------------------------------------------------------------------------- #
# Fast path: trust KITTI's published rectification                             #
# --------------------------------------------------------------------------- #

def rectification_from_kitti(calib) -> StereoRectification:
    """Build the same StereoRectification using the values KITTI publishes.

    KITTI's calib_cam_to_cam.txt already contains R_rect_xx and P_rect_xx;
    these are the values their devkit uses to produce the synced+rectified
    image streams. Using them guarantees the rectified images match KITTI's
    ground truth exactly (within a remap precision).

    Use this path for the demo. Use build_rectification() to demonstrate the
    from-scratch derivation in the technical report.
    """
    W, H = calib.S_L
    fx_new = calib.P_L_rect[0, 0]
    fy_new = calib.P_L_rect[1, 1]
    cx_new = calib.P_L_rect[0, 2]
    cy_new = calib.P_L_rect[1, 2]
    K_new = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]], dtype=np.float64)

    map_Lx, map_Ly = _build_remap(K_new, calib.R_L, calib.K_L_unrect, calib.D_L, (W, H))
    map_Rx, map_Ry = _build_remap(K_new, calib.R_R, calib.K_R_unrect, calib.D_R, (W, H))

    baseline = float(np.linalg.norm(calib.T_LR))
    return StereoRectification(
        map_Lx=map_Lx, map_Ly=map_Ly, map_Rx=map_Rx, map_Ry=map_Ry,
        K_rect_L=K_new, K_rect_R=K_new, baseline=baseline, out_size=(W, H),
    )


def passthrough_rectification(calib) -> StereoRectification:
    """For already-rectified input (KITTI sync drives): identity remap at S_rect.

    The KITTI sync drives ship images that are already rectified using the same
    R_rect / P_rect we'd derive ourselves. In that case there's nothing left to
    warp — we just pass the image through and use P_rect's top-left 3x3 as the
    rectified intrinsic. This lets the bonus run end-to-end on sync drives
    without needing the bigger _extract archives.
    """
    W, H = calib.S_rect_L
    fx_new = calib.P_L_rect[0, 0]
    fy_new = calib.P_L_rect[1, 1]
    cx_new = calib.P_L_rect[0, 2]
    cy_new = calib.P_L_rect[1, 2]
    K_new = np.array([[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]], dtype=np.float64)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    baseline = float(np.linalg.norm(calib.T_LR))
    return StereoRectification(
        map_Lx=map_x, map_Ly=map_y, map_Rx=map_x.copy(), map_Ry=map_y.copy(),
        K_rect_L=K_new, K_rect_R=K_new, baseline=baseline, out_size=(W, H),
    )


def apply(img_L: np.ndarray, img_R: np.ndarray, rect: StereoRectification) -> tuple[np.ndarray, np.ndarray]:
    """Warp raw stereo images through the precomputed remap LUTs."""
    rL = cv2.remap(img_L, rect.map_Lx, rect.map_Ly, interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)
    rR = cv2.remap(img_R, rect.map_Rx, rect.map_Ry, interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)
    return rL, rR

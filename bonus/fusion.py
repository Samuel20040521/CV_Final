"""Multi-frame TSDF fusion of per-frame depth into a global mesh.

We use Open3D's `ScalableTSDFVolume` — a sparse voxel hash that scales to large
outdoor scenes (KITTI sequences span tens of metres). Each frame contributes
its rectified left RGB image + metric depth map, transformed into world frame
via the camera pose:

    T_world_cam = T_world_imu · T_imu_cam

where T_world_imu comes from KITTI's oxts records and T_imu_cam from the
calibration cascade (imu→velo→cam→rect).

TSDF integration weights samples by viewing angle, so noisy depth at grazing
angles contributes less and dense same-surface samples reinforce each other.
This averages out per-frame stereo noise into a coherent reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class TSDFConfig:
    voxel_size: float = 0.08          # metres
    sdf_trunc: float = 0.32           # 4x voxel_size, a common rule
    depth_trunc: float = 50.0         # m, drop samples beyond this
    depth_scale: float = 1000.0       # depth values in millimetres for Open3D
    color_type: o3d.pipelines.integration.TSDFVolumeColorType = (
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )


def make_volume(cfg: TSDFConfig = TSDFConfig()) -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.sdf_trunc,
        color_type=cfg.color_type,
    )


def integrate_frame(
    vol: o3d.pipelines.integration.ScalableTSDFVolume,
    color_bgr: np.ndarray,
    depth_metres: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    cfg: TSDFConfig,
) -> None:
    """Add one rectified RGB+depth frame to the global TSDF volume."""
    H, W = depth_metres.shape
    color_rgb = color_bgr[..., ::-1]
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_rgb))
    depth_mm = (depth_metres * cfg.depth_scale).astype(np.uint16)
    depth_o3d = o3d.geometry.Image(depth_mm)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=cfg.depth_scale,
        depth_trunc=cfg.depth_trunc,
        convert_rgb_to_intensity=False,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    # Open3D wants extrinsic = T_cam_world (the inverse of pose).
    extrinsic = np.linalg.inv(T_world_cam)
    vol.integrate(rgbd, intrinsic, extrinsic)


def extract_mesh(vol: o3d.pipelines.integration.ScalableTSDFVolume) -> o3d.geometry.TriangleMesh:
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

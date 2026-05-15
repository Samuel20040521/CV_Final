"""Offscreen flythrough rendering of the fused mesh into an mp4.

Uses Open3D's headless OffscreenRenderer (works without a display server).
Camera path is a smooth orbit around the trajectory midpoint with the camera
slowly descending toward the scene — gives an unambiguously 3D-looking demo
without requiring full free-viewpoint control.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import open3d as o3d


def _camera_path(center: np.ndarray, n_frames: int, radius: float = 25.0, height: float = 10.0) -> list[np.ndarray]:
    """Return n_frames world-space camera positions on a circle around `center`."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    positions = []
    for a in angles:
        # KITTI world: x = east, y = north, z = up.
        positions.append(center + np.array([radius * np.cos(a), radius * np.sin(a), height]))
    return positions


def render_pointcloud_flythrough(
    pcd: o3d.geometry.PointCloud,
    out_path: Path,
    n_frames: int = 240,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    radius: float = 12.0,
    cam_height: float = 4.0,
    point_size: float = 2.0,
) -> Path:
    """Orbit-render a colored point cloud (used for the single-frame demo)."""
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.05, 0.05, 0.08, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    scene.add_geometry("pcd", pcd, mat)

    positions = _camera_path(center, n_frames, radius=radius, height=cam_height)

    frames = []
    for pos in positions:
        renderer.scene.camera.look_at(center.tolist(), pos.tolist(), [0.0, -1.0, 0.0])
        img = np.asarray(renderer.render_to_image())
        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps, codec="libx264")
    return out_path


def render_flythrough(
    mesh: o3d.geometry.TriangleMesh,
    out_path: Path,
    n_frames: int = 240,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    radius: float = 25.0,
    cam_height: float = 10.0,
) -> Path:
    """Render an orbit flythrough of `mesh` to an mp4 at `out_path`."""
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.05, 0.05, 0.08, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    positions = _camera_path(center, n_frames, radius=radius, height=cam_height)

    frames = []
    for pos in positions:
        renderer.scene.camera.look_at(center.tolist(), pos.tolist(), [0.0, 0.0, 1.0])
        img = np.asarray(renderer.render_to_image())
        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps, codec="libx264")
    return out_path

"""Single-frame bonus demo.

One KITTI stereo pair → rectify → computeDisp → colored point cloud →
orbit-render the cloud to an mp4. No multi-frame fusion, no poses.

Useful as a fast smoke demo and as the "the depth is real 3D" demo for the
report — the orbiting viewpoint makes parallax obvious without needing the
TSDF pipeline.

Usage:
    uv run python -m bonus.single_frame \
        --kitti-root data/kitti_raw \
        --date 2011_09_26 --drive 0005 --frame 0 \
        --max-disp 96 \
        --out out/single_frame.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from stereo_matching import computeDisp
from . import depth as depth_mod
from . import kitti
from . import rectify as rect_mod
from . import render


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-frame stereo → point cloud demo")
    p.add_argument("--kitti-root", required=True, type=Path)
    p.add_argument("--date", required=True)
    p.add_argument("--drive", required=True)
    p.add_argument("--frame", type=int, default=0, help="frame index (0-based)")
    p.add_argument("--max-disp", type=int, default=96)
    p.add_argument("--rectify", choices=["auto", "kitti", "scratch", "passthrough"], default="auto")
    p.add_argument("--max-depth", type=float, default=40.0,
                   help="drop points farther than this; stereo gets unreliable past ~30m on KITTI")
    p.add_argument("--out", required=True, type=Path, help="output mp4 path")
    p.add_argument("--ply", type=Path, default=None,
                   help="optional: also save the point cloud as a .ply for Open3D viewing")
    p.add_argument("--disp-png", type=Path, default=None,
                   help="optional: save the disparity map as a visualisation PNG")
    p.add_argument("--n-render-frames", type=int, default=180)
    p.add_argument("--orbit-radius", type=float, default=12.0)
    p.add_argument("--orbit-height", type=float, default=4.0)
    p.add_argument("--point-size", type=float, default=2.0)
    return p.parse_args()


def _build_rectification(args, calib, probe_size):
    is_rectified = probe_size == calib.S_rect_L
    mode = args.rectify
    if mode == "auto":
        mode = "passthrough" if is_rectified else "kitti"
    elif mode in ("kitti", "scratch") and is_rectified:
        print(f"[warn] --rectify {mode} on rectified input; using passthrough")
        mode = "passthrough"
    elif mode == "passthrough" and not is_rectified:
        raise SystemExit("--rectify passthrough needs already-rectified input")

    if mode == "passthrough":
        return rect_mod.passthrough_rectification(calib), mode
    if mode == "kitti":
        return rect_mod.rectification_from_kitti(calib), mode
    R_rel = calib.R_R @ calib.R_L.T
    return rect_mod.build_rectification(
        K_L=calib.K_L_unrect, K_R=calib.K_R_unrect, R=R_rel, T=calib.T_LR,
        image_size=calib.S_L, D_L=calib.D_L, D_R=calib.D_R,
    ), mode


def main() -> None:
    args = parse_args()
    date_dir = args.kitti_root / args.date
    drive_dir = date_dir / f"{args.date}_drive_{args.drive}_sync"
    if not drive_dir.exists():
        raise FileNotFoundError(drive_dir)

    print(f"[setup] loading calibration from {date_dir}")
    calib = kitti.load_calibration(date_dir)
    pairs = kitti.list_pairs(drive_dir)
    if not 0 <= args.frame < len(pairs):
        raise SystemExit(f"--frame {args.frame} out of range (have {len(pairs)} frames)")

    L_path, R_path = pairs[args.frame]
    print(f"[setup] frame {args.frame}: {L_path.name}")
    Il = cv2.imread(str(L_path))
    Ir = cv2.imread(str(R_path))
    if Il is None or Ir is None:
        raise SystemExit(f"failed to read frame {args.frame}")

    rect, mode = _build_rectification(args, calib, (Il.shape[1], Il.shape[0]))
    print(f"[setup] rectification: {mode}; fx={rect.K_rect_L[0,0]:.1f}, "
          f"baseline={rect.baseline:.4f} m")

    rL, rR = rect_mod.apply(Il, Ir, rect)

    print(f"[run] computeDisp (max_disp={args.max_disp})")
    t0 = time.time()
    disp = computeDisp(rL, rR, args.max_disp).astype(np.float32)
    print(f"[run] disparity in {time.time()-t0:.2f}s; "
          f"valid={(disp > 0).mean()*100:.1f}%, range=[{disp.min()}, {disp.max()}]")

    if args.disp_png is not None:
        args.disp_png.parent.mkdir(parents=True, exist_ok=True)
        vis = (disp * (255.0 / max(1, args.max_disp))).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(args.disp_png), vis)
        print(f"[save] disparity → {args.disp_png}")

    depth = depth_mod.disparity_to_depth(disp, fx=rect.K_rect_L[0, 0], baseline=rect.baseline)
    pts, cols = depth_mod.depth_to_points(depth, rL, rect.K_rect_L, max_depth=args.max_depth)
    print(f"[points] {len(pts):,} points (depth ≤ {args.max_depth} m)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    if args.ply is not None:
        args.ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.ply), pcd)
        print(f"[save] point cloud → {args.ply}")

    print(f"[render] writing flythrough to {args.out}")
    render.render_pointcloud_flythrough(
        pcd, args.out,
        n_frames=args.n_render_frames,
        radius=args.orbit_radius,
        cam_height=args.orbit_height,
        point_size=args.point_size,
    )
    print("[done]")


if __name__ == "__main__":
    main()

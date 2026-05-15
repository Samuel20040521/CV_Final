"""End-to-end bonus pipeline: KITTI sequence → fused 3D mesh → flythrough mp4.

Usage:
    uv run python -m bonus.run_bonus \
        --kitti-root /path/to/kitti_raw \
        --date 2011_09_26 \
        --drive 0005 \
        --start 0 --end 100 --stride 2 \
        --max-disp 96 \
        --rectify kitti \
        --out out/bonus_2011_09_26_drive_0005.mp4

The pipeline reuses stereo_matching.computeDisp without modification, so the
main assignment's 80% performance score is unaffected by anything in here.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from stereo_matching import computeDisp
from . import depth as depth_mod
from . import fusion
from . import kitti
from . import rectify as rect_mod
from . import render


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bonus stereo pipeline on KITTI raw")
    p.add_argument("--kitti-root", required=True, type=Path,
                   help="path to KITTI raw root (contains date directories)")
    p.add_argument("--date", required=True, help="e.g. 2011_09_26")
    p.add_argument("--drive", required=True, help="e.g. 0005 (the 4-digit drive id)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="exclusive; default = all frames")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-disp", type=int, default=96,
                   help="max disparity search range (KITTI baseline is ~0.54m so 96 covers ~1.5m near)")
    p.add_argument("--rectify", choices=["auto", "kitti", "scratch", "passthrough"], default="auto",
                   help="auto: detect from first frame size (sync drives are already rectified, "
                        "extract drives are not). kitti: use published R_rect/P_rect on raw input. "
                        "scratch: from-scratch rectification on raw input. passthrough: skip remap "
                        "(input already rectified).")
    p.add_argument("--out", required=True, type=Path, help="output mp4 path")
    p.add_argument("--voxel", type=float, default=0.08, help="TSDF voxel size in metres")
    p.add_argument("--orbit-radius", type=float, default=25.0)
    p.add_argument("--orbit-height", type=float, default=10.0)
    p.add_argument("--n-render-frames", type=int, default=240)
    p.add_argument("--debug-disp-dir", type=Path, default=None,
                   help="if set, save the disparity map of each processed frame here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    date_dir = args.kitti_root / args.date
    drive_dir = date_dir / f"{args.date}_drive_{args.drive}_sync"
    if not drive_dir.exists():
        raise FileNotFoundError(f"drive directory not found: {drive_dir}")

    print(f"[setup] loading calibration from {date_dir}")
    calib = kitti.load_calibration(date_dir)

    pairs = kitti.list_pairs(drive_dir)
    end = args.end if args.end is not None else len(pairs)
    frame_idx = range(args.start, end, args.stride)
    print(f"[setup] {len(pairs)} frames available; processing {len(frame_idx)} (stride {args.stride})")

    print(f"[setup] loading {len(frame_idx)} IMU/GPS poses")
    poses_world_imu = kitti.load_poses(drive_dir, range(args.start, end, 1))
    T_imu_to_cam = kitti.load_imu_to_cam(date_dir)
    T_cam_to_imu = np.linalg.inv(T_imu_to_cam)

    # Detect input image dimensions to pick the right rectification mode.
    probe_L, _ = pairs[args.start]
    probe_img = cv2.imread(str(probe_L))
    probe_h, probe_w = probe_img.shape[:2]
    is_rectified = (probe_w, probe_h) == calib.S_rect_L
    print(f"[setup] input frame size {probe_w}x{probe_h}  "
          f"(unrectified={calib.S_L}, rectified={calib.S_rect_L}) → "
          f"{'pre-rectified' if is_rectified else 'raw'}")

    mode = args.rectify
    if mode == "auto":
        mode = "passthrough" if is_rectified else "kitti"
        print(f"[setup] --rectify auto → {mode}")
    elif mode in ("kitti", "scratch") and is_rectified:
        print(f"[warn] --rectify {mode} requested but input is already rectified; "
              f"falling back to passthrough to avoid double-rectifying")
        mode = "passthrough"
    elif mode == "passthrough" and not is_rectified:
        raise SystemExit("--rectify passthrough requires already-rectified input "
                         f"(got {probe_w}x{probe_h}, expected {calib.S_rect_L})")

    print(f"[setup] building rectification ({mode})")
    if mode == "passthrough":
        rect = rect_mod.passthrough_rectification(calib)
    elif mode == "kitti":
        rect = rect_mod.rectification_from_kitti(calib)
    else:
        # From-scratch path: feed it the raw intrinsics and the relative pose
        # of the right cam in the left cam frame (R_R · R_L⁻¹, T_LR).
        R_rel = calib.R_R @ calib.R_L.T
        rect = rect_mod.build_rectification(
            K_L=calib.K_L_unrect, K_R=calib.K_R_unrect,
            R=R_rel, T=calib.T_LR,
            image_size=calib.S_L,
            D_L=calib.D_L, D_R=calib.D_R,
        )
    fx = rect.K_rect_L[0, 0]
    print(f"[setup] rectified fx={fx:.1f}, baseline={rect.baseline:.4f} m")

    cfg = fusion.TSDFConfig(voxel_size=args.voxel)
    vol = fusion.make_volume(cfg)

    if args.debug_disp_dir is not None:
        args.debug_disp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] integrating {len(frame_idx)} frames")
    t0 = time.time()
    for k, i in enumerate(tqdm(frame_idx)):
        L_path, R_path = pairs[i]
        Il = cv2.imread(str(L_path))
        Ir = cv2.imread(str(R_path))
        if Il is None or Ir is None:
            print(f"[warn] skipping frame {i} (missing image)")
            continue

        rL, rR = rect_mod.apply(Il, Ir, rect)
        disp = computeDisp(rL, rR, args.max_disp).astype(np.float32)
        depth = depth_mod.disparity_to_depth(disp, fx=fx, baseline=rect.baseline)

        if args.debug_disp_dir is not None:
            cv2.imwrite(str(args.debug_disp_dir / f"disp_{i:06d}.png"),
                        (disp * (255.0 / max(1, args.max_disp))).astype(np.uint8))

        # World-frame camera pose for this frame
        T_world_imu = poses_world_imu[i - args.start]
        T_world_cam = T_world_imu @ T_cam_to_imu

        fusion.integrate_frame(vol, rL, depth, rect.K_rect_L, T_world_cam, cfg)
    dt = time.time() - t0
    print(f"[run] integration done in {dt:.1f}s ({dt / max(1, len(frame_idx)):.2f}s/frame)")

    print("[mesh] extracting triangle mesh")
    mesh = fusion.extract_mesh(vol)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print(f"[mesh] vertices={n_v}, triangles={n_t}")
    if n_v == 0:
        raise RuntimeError("empty mesh — check disparity/pose alignment")

    print(f"[render] writing flythrough to {args.out}")
    render.render_flythrough(
        mesh, args.out,
        n_frames=args.n_render_frames,
        radius=args.orbit_radius,
        cam_height=args.orbit_height,
    )
    print("[done]")


if __name__ == "__main__":
    main()

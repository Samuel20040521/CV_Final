"""End-to-end bonus pipeline: TartanAir sequence → fused 3D mesh → flythrough mp4.

Reuses ``stereo_matching.computeDisp`` (v6) unchanged. Adds a ``--use-gt-depth``
flag that bypasses the matcher and integrates TartanAir's ground-truth depth
maps instead — this gives the pipeline's *upper bound* and is the right
reference video to put in the report next to the v6 result.

Usage::

    # v6 prediction (main demo)
    uv run python -m bonus.run_bonus_tartanair --traj P000 --end 100 \\
        --out out/tartanair_japanesealley_P000_v6.mp4

    # GT-depth reference (pipeline upper bound)
    uv run python -m bonus.run_bonus_tartanair --traj P000 --end 100 \\
        --use-gt-depth --out out/tartanair_japanesealley_P000_gt.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from computeDisp import computeDisp
from . import depth as depth_mod
from . import fusion
from . import render
from . import tartanair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bonus stereo+fusion pipeline on TartanAir")
    p.add_argument("--root", default=Path("data/tartanair_raw"), type=Path)
    p.add_argument("--scene", default="japanesealley")
    p.add_argument("--level", default="Hard", choices=["Easy", "Hard"])
    p.add_argument("--traj", default="P000")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100, help="exclusive")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-disp", type=int, default=64,
                   help="disparity search range; P000-style 'far' trajectories peak ~33")
    p.add_argument("--use-gt-depth", action="store_true",
                   help="integrate TartanAir GT depth instead of v6 prediction "
                        "(produces the pipeline upper-bound reference demo)")
    p.add_argument("--voxel", type=float, default=0.05,
                   help="TSDF voxel size in metres; TartanAir alleys are ~10-20 m wide "
                        "so 0.05 keeps walls sharp without exploding voxel count")
    p.add_argument("--depth-trunc", type=float, default=15.0,
                   help="drop depth samples beyond this distance; classical stereo "
                        "depth variance scales as Z²/(fB) so far points add mostly noise")
    p.add_argument("--render-upscale", type=int, default=2,
                   help="render at this multiple of the input 640x480 resolution; "
                        "keeps aspect ratio (anisotropic scaling would break the intrinsic)")
    p.add_argument("--render-fps", type=int, default=15,
                   help="output mp4 fps; defaults to half of TartanAir's 30Hz so each "
                        "frame is on screen long enough to inspect")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--debug-disp-dir", type=Path, default=None,
                   help="if set, save each frame's disparity map here as PNG")
    return p.parse_args()


def _prepare_depth(depth: np.ndarray, depth_trunc: float) -> np.ndarray:
    """Return a depth map with sky / out-of-range pixels zeroed.

    Open3D's TSDF treats 0-depth as invalid (skipped), while it *clips* values
    above ``depth_trunc`` to the trunc value — which would inject phantom
    surfaces at the truncation distance. Zeroing both invalid bands avoids that.
    """
    out = depth.astype(np.float32, copy=True)
    out[out >= tartanair.SKY_DEPTH_THRESHOLD] = 0.0
    out[out > depth_trunc] = 0.0
    return out


def main() -> None:
    args = parse_args()
    traj_dir = args.root / args.scene / args.level / args.traj
    if not traj_dir.is_dir():
        raise SystemExit("[error] not a directory: {}".format(traj_dir))

    K = tartanair.get_intrinsics()
    fx = K[0, 0]
    baseline = tartanair.BASELINE_M
    indices = range(args.start, args.end, args.stride)

    print("[setup] {}/{}/{} frames {}..{} stride={}".format(
        args.scene, args.level, args.traj, args.start, args.end - 1, args.stride))
    print("[setup] fx={:.1f}  baseline={:.4f} m".format(fx, baseline))

    frames = tartanair.list_frames(traj_dir, indices)
    poses = tartanair.load_poses(traj_dir, indices)
    if len(frames) != len(poses):
        raise RuntimeError("frame/pose count mismatch: {} vs {}".format(len(frames), len(poses)))
    print("[setup] {} frames + {} poses loaded".format(len(frames), len(poses)))

    cfg = fusion.TSDFConfig(
        voxel_size=args.voxel,
        sdf_trunc=4 * args.voxel,
        depth_trunc=args.depth_trunc,
    )
    vol = fusion.make_volume(cfg)

    if args.debug_disp_dir is not None:
        args.debug_disp_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "GT depth" if args.use_gt_depth else "v6 (max_disp={})".format(args.max_disp)
    print("[run] integrating {} frames  mode={}".format(len(frames), mode_label))
    t0 = time.time()
    for k, (frame, T_world_cam) in enumerate(zip(tqdm(frames), poses)):
        Il = cv2.imread(str(frame.image_left))
        if Il is None:
            print("[warn] skipping frame {}: cannot read {}".format(k, frame.image_left))
            continue

        if args.use_gt_depth:
            depth = np.load(frame.depth_left)
        else:
            Ir = cv2.imread(str(frame.image_right))
            disp = computeDisp(Il, Ir, args.max_disp).astype(np.float32)
            depth = depth_mod.disparity_to_depth(disp, fx=fx, baseline=baseline)
            if args.debug_disp_dir is not None:
                idx = args.start + k * args.stride
                cv2.imwrite(
                    str(args.debug_disp_dir / "disp_{:06d}.png".format(idx)),
                    (disp * (255.0 / max(1, args.max_disp))).astype(np.uint8),
                )

        depth = _prepare_depth(depth, args.depth_trunc)
        fusion.integrate_frame(vol, Il, depth, K, T_world_cam, cfg)

    dt = time.time() - t0
    print("[run] integration done in {:.1f}s ({:.2f}s/frame)".format(
        dt, dt / max(1, len(frames))))

    print("[mesh] extracting triangle mesh")
    mesh = fusion.extract_mesh(vol)
    n_v = len(mesh.vertices)
    n_t = len(mesh.triangles)
    print("[mesh] vertices={}  triangles={}".format(n_v, n_t))
    if n_v == 0:
        raise SystemExit("[error] empty mesh — check pose/depth alignment")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("[render] follow-trajectory walkthrough → {}".format(args.out))
    iw, ih = tartanair.IMAGE_HW[1], tartanair.IMAGE_HW[0]
    render.render_mesh_follow_trajectory(
        mesh, poses, K, (iw, ih), args.out,
        fps=args.render_fps,
        upscale=args.render_upscale,
    )
    print("[done]")


if __name__ == "__main__":
    main()

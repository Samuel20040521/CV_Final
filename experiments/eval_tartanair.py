"""Evaluate v6 stereo matcher against TartanAir GT depth.

For each frame we run ``computeDisp(Il, Ir, max_disp)``, convert GT depth to
GT disparity via the camera's fx and baseline, and report a Middlebury-style
BPR (% pixels with |D_pred - D_gt| > threshold) over the valid mask.

Valid mask excludes:
  * sky / inf pixels (TartanAir marks them with depth ≈ 65504 = float16 max)
  * pixels whose GT disparity exceeds ``--max-disp`` (the matcher cannot
    possibly recover them, so counting them as wrong would only measure
    the search-range choice)

TartanAir camera (all frames): fx = fy = 320, baseline = 0.25 m, 640×480.

Usage::

    uv run python -m experiments.eval_tartanair                    # P000 first 100
    uv run python -m experiments.eval_tartanair --traj P001 --end 50
    uv run python -m experiments.eval_tartanair --max-disp 96 --threshold 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from computeDisp import computeDisp


# TartanAir intrinsics (constant for the whole dataset).
FX = 320.0
BASELINE_M = 0.25
# Sky / inf marker — observed maximum value is 65504.0 (float16 max).
SKY_THRESHOLD = 65000.0


def _load_pair(traj_dir: Path, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Il_bgr, Ir_bgr, depth_left_m) for frame ``idx``.

    Frame stems are 6-digit zero-padded (``000042_left.png`` etc.).
    """
    stem = "{:06d}".format(idx)
    il = cv2.imread(str(traj_dir / "image_left" / "{}_left.png".format(stem)))
    ir = cv2.imread(str(traj_dir / "image_right" / "{}_right.png".format(stem)))
    depth = np.load(traj_dir / "depth_left" / "{}_left_depth.npy".format(stem))
    if il is None or ir is None:
        raise FileNotFoundError("missing image for frame {} in {}".format(idx, traj_dir))
    return il, ir, depth


def _bpr(d_pred: np.ndarray, d_gt: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Bad-pixel ratio over ``mask`` with the given disparity threshold."""
    if mask.sum() == 0:
        return float("nan")
    bad = np.abs(d_pred.astype(np.float32) - d_gt) > threshold
    return float(bad[mask].mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate computeDisp on TartanAir GT depth.")
    p.add_argument("--root", default=Path("data/tartanair_raw"), type=Path)
    p.add_argument("--scene", default="japanesealley")
    p.add_argument("--level", default="Hard", choices=["Easy", "Hard"])
    p.add_argument("--traj", default="P000")
    p.add_argument("--start", default=0, type=int, help="first frame index (inclusive)")
    p.add_argument("--end", default=100, type=int, help="last frame index (exclusive)")
    p.add_argument("--max-disp", default=64, type=int)
    p.add_argument("--threshold", default=1.0, type=float, help="disparity error threshold (pixels)")
    p.add_argument("--verbose", action="store_true", help="print per-frame BPR")
    args = p.parse_args()

    traj_dir = args.root / args.scene / args.level / args.traj
    if not traj_dir.is_dir():
        raise SystemExit("[error] not a directory: {}".format(traj_dir))

    fx_b = FX * BASELINE_M  # disparity = fx_b / depth

    per_frame_bpr: List[float] = []
    per_frame_time: List[float] = []
    per_frame_valid_frac: List[float] = []

    iterator = range(args.start, args.end)
    pbar = tqdm(iterator, desc="{}/{}/{}".format(args.scene, args.level, args.traj))
    for i in pbar:
        il, ir, depth = _load_pair(traj_dir, i)

        d_gt = fx_b / np.clip(depth, 1e-6, None)  # safe divide

        t0 = time.perf_counter()
        d_pred = computeDisp(il, ir, args.max_disp)
        dt = time.perf_counter() - t0

        valid = (depth < SKY_THRESHOLD) & (d_gt < args.max_disp)
        bpr = _bpr(d_pred, d_gt, valid, args.threshold)
        per_frame_bpr.append(bpr)
        per_frame_time.append(dt)
        per_frame_valid_frac.append(float(valid.mean()))

        if args.verbose:
            tqdm.write("frame {:04d}  BPR={:.2%}  t={:.2f}s  valid={:.1%}".format(
                i, bpr, dt, valid.mean()))
        pbar.set_postfix(bpr="{:.2%}".format(bpr), t="{:.2f}s".format(dt))

    arr_bpr = np.asarray(per_frame_bpr)
    arr_t = np.asarray(per_frame_time)
    arr_v = np.asarray(per_frame_valid_frac)

    print()
    print("=" * 60)
    print("Scene:      {}/{}/{}  frames {}..{}".format(
        args.scene, args.level, args.traj, args.start, args.end - 1))
    print("Settings:   max_disp={}  threshold={:.1f}px".format(args.max_disp, args.threshold))
    print("-" * 60)
    print("BPR  mean={:.2%}  median={:.2%}  worst={:.2%}  best={:.2%}".format(
        np.nanmean(arr_bpr), np.nanmedian(arr_bpr), np.nanmax(arr_bpr), np.nanmin(arr_bpr)))
    print("Time mean={:.2f}s  total={:.1f}s".format(arr_t.mean(), arr_t.sum()))
    print("Valid-mask coverage mean={:.1%}".format(arr_v.mean()))
    print("=" * 60)


if __name__ == "__main__":
    main()

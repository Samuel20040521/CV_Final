"""Hyperparameter sweep harness for the basic-score pipeline.

Calls the stereo_matching internals directly (bypassing the public computeDisp
shim) so each (radius, eps, wmf_r, lrc_th) tuple can be evaluated without
mutating the production code. Reports BPR + time per image, and an overall
4-image average. Optimization target: minimize average BPR while keeping every
image safely below its threshold.

Run:
    uv run python -m scripts.sweep
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from stereo_matching.aggregation import aggregate
from stereo_matching.cost import census_transform, hamming_cost_volume
from stereo_matching.optimization import winner_take_all
from stereo_matching.refinement import hole_fill, lr_consistency, weighted_median

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"

CONFIG = {
    "Tsukuba": (15, 16, 8.0),   # (max_disp, scale_factor, threshold_pct)
    "Venus":   (20, 8, 5.0),
    "Teddy":   (60, 4, 18.0),
    "Cones":   (60, 4, 15.0),
}


def run_pipeline(
    Il: np.ndarray, Ir: np.ndarray, max_disp: int,
    gf_radius: int, gf_eps: float, wmf_radius: int, lrc_thresh: int,
) -> np.ndarray:
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    census_L = census_transform(gray_L)
    census_R = census_transform(gray_R)
    cost_L = hamming_cost_volume(census_L, census_R, max_disp, direction="L")
    cost_R = hamming_cost_volume(census_L, census_R, max_disp, direction="R")

    cost_L = aggregate(cost_L, Il, radius=gf_radius, eps=gf_eps)
    cost_R = aggregate(cost_R, Ir, radius=gf_radius, eps=gf_eps)

    D_L = winner_take_all(cost_L)
    D_R = winner_take_all(cost_R)

    valid = lr_consistency(D_L, D_R, threshold=lrc_thresh)
    filled = hole_fill(D_L, valid, max_disp)
    return weighted_median(filled, Il, radius=wmf_radius).astype(np.uint8)


def evaluate_bpr(disp_pred: np.ndarray, disp_gt: np.ndarray, scale_factor: int, threshold: float = 1.0) -> float:
    """Replicates eval.py's evaluate() exactly, but vectorized."""
    disp_p = np.int32(disp_pred * scale_factor) // 1  # eval does u8 cast then int32 → noop here
    disp_p = np.int32(disp_p / scale_factor)
    disp_g = np.int32(disp_gt / scale_factor)
    mask = disp_g > 0
    if mask.sum() == 0:
        return 0.0
    errors = np.abs(disp_g[mask] - disp_p[mask]) > threshold
    return float(errors.sum()) / float(mask.sum())


def _load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    Il = cv2.imread(str(TESTDATA / name / "img_left.png"))
    Ir = cv2.imread(str(TESTDATA / name / "img_right.png"))
    gt = cv2.imread(str(TESTDATA / name / "disp_gt.png"), -1)
    max_disp, scale, thresh = CONFIG[name]
    return Il, Ir, gt, max_disp, scale, thresh


def run_config(gf_radius: int, gf_eps: float, wmf_radius: int, lrc_thresh: int,
               datasets: Iterable[str]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name in datasets:
        Il, Ir, gt, max_disp, scale, thresh = _load_dataset(name)
        t0 = time.time()
        pred = run_pipeline(Il, Ir, max_disp, gf_radius, gf_eps, wmf_radius, lrc_thresh)
        dt = time.time() - t0
        bpr = evaluate_bpr(pred, gt, scale) * 100
        results[name] = {"bpr": bpr, "time": dt, "threshold": thresh, "margin": thresh - bpr}
    return results


def fmt_row(name: str, r: Dict[str, float]) -> str:
    pass_ = "✓" if r["bpr"] < r["threshold"] else "✗"
    return f"  {name:8s} BPR={r['bpr']:5.2f}% (th<{r['threshold']:4.1f}) margin={r['margin']:+5.2f} time={r['time']:.2f}s {pass_}"


def summarize(results: Dict[str, Dict[str, float]]) -> Tuple[float, bool]:
    bprs = [r["bpr"] for r in results.values()]
    avg = sum(bprs) / len(bprs)
    all_pass = all(r["bpr"] < r["threshold"] for r in results.values())
    return avg, all_pass


def sweep_axis(name: str, configs: List[Dict], datasets: Iterable[str]) -> List[Tuple[Dict, float, bool, Dict]]:
    print(f"\n{'='*70}")
    print(f"Sweep: {name}  ({len(configs)} configs)")
    print('='*70)
    rows = []
    for cfg in configs:
        results = run_config(
            gf_radius=cfg["gf_r"], gf_eps=cfg["gf_eps"],
            wmf_radius=cfg["wmf_r"], lrc_thresh=cfg["lrc"],
            datasets=datasets,
        )
        avg, all_pass = summarize(results)
        marker = "PASS" if all_pass else "FAIL"
        label = f"gf_r={cfg['gf_r']:2d} gf_eps={cfg['gf_eps']:.0e} wmf_r={cfg['wmf_r']:2d} lrc={cfg['lrc']}"
        print(f"[{marker}] {label}  avg={avg:5.2f}%")
        for ds, r in results.items():
            print(fmt_row(ds, r))
        rows.append((cfg, avg, all_pass, results))
    return rows


def best_config(rows: List[Tuple[Dict, float, bool, Dict]]) -> Tuple[Dict, float, Dict]:
    passing = [(cfg, avg, results) for cfg, avg, ok, results in rows if ok]
    if not passing:
        raise SystemExit("No passing config in this axis — aborting.")
    return min(passing, key=lambda x: x[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(CONFIG.keys()))
    parser.add_argument("--quick", action="store_true", help="smaller grid for fast iteration")
    args = parser.parse_args()

    datasets = args.datasets

    # Baseline reference (current defaults)
    base = {"gf_r": 11, "gf_eps": 1e-4, "wmf_r": 17, "lrc": 1}
    base_results = run_config(base["gf_r"], base["gf_eps"], base["wmf_r"], base["lrc"], datasets)
    base_avg, _ = summarize(base_results)
    base_teddy = base_results.get("Teddy", {}).get("bpr", 11.85)
    print(f"\n[BASELINE] {base}  avg={base_avg:.2f}%")
    for ds, r in base_results.items():
        print(fmt_row(ds, r))

    # Round 1: guided filter radius + eps
    if args.quick:
        gf_rs = [9, 11, 13]
        gf_epses = [1e-4, 1e-3]
    else:
        gf_rs = [7, 9, 11, 13, 15]
        gf_epses = [1e-4, 5e-4, 1e-3]
    round1 = [
        {"gf_r": r, "gf_eps": e, "wmf_r": base["wmf_r"], "lrc": base["lrc"]}
        for r in gf_rs for e in gf_epses
    ]
    rows1 = sweep_axis("Guided filter (radius, eps)", round1, datasets)
    cfg1, avg1, res1 = best_config(rows1)
    teddy1 = res1.get("Teddy", {}).get("bpr", base_teddy)
    print(f"\n[ROUND 1 BEST] {cfg1}  avg={avg1:.2f}% (vs base {base_avg:.2f}%)  teddy={teddy1:.2f}")

    # Round 2: WMF radius, locked to round 1 best
    if args.quick:
        wmf_rs = [13, 15, 17, 19]
    else:
        wmf_rs = [11, 13, 15, 17, 19, 21]
    round2 = [
        {"gf_r": cfg1["gf_r"], "gf_eps": cfg1["gf_eps"], "wmf_r": w, "lrc": cfg1["lrc"]}
        for w in wmf_rs
    ]
    rows2 = sweep_axis("WMF radius", round2, datasets)
    cfg2, avg2, res2 = best_config(rows2)
    teddy2 = res2.get("Teddy", {}).get("bpr", base_teddy)
    print(f"\n[ROUND 2 BEST] {cfg2}  avg={avg2:.2f}% (vs round1 {avg1:.2f}%)  teddy={teddy2:.2f}")

    # Round 3: LRC threshold
    round3 = [
        {"gf_r": cfg2["gf_r"], "gf_eps": cfg2["gf_eps"], "wmf_r": cfg2["wmf_r"], "lrc": t}
        for t in [1, 2]
    ]
    rows3 = sweep_axis("LRC threshold", round3, datasets)
    cfg3, avg3, res3 = best_config(rows3)
    teddy3 = res3.get("Teddy", {}).get("bpr", base_teddy)
    print(f"\n[ROUND 3 BEST] {cfg3}  avg={avg3:.2f}% (vs round2 {avg2:.2f}%)  teddy={teddy3:.2f}")

    # Anti-overfit guard: accept only if avg improves > 0.5 pp AND Teddy doesn't regress > 1 pp
    delta = base_avg - avg3
    teddy_regress = teddy3 - base_teddy
    print(f"\n{'='*70}")
    print("Final decision")
    print('='*70)
    print(f"Baseline avg = {base_avg:.2f}%   Best avg = {avg3:.2f}%   Δ = {delta:+.2f} pp")
    print(f"Baseline Teddy = {base_teddy:.2f}%   Best Teddy = {teddy3:.2f}%   regression = {teddy_regress:+.2f} pp")
    if delta > 0.5 and teddy_regress <= 1.0:
        print(f"ACCEPT: avg gain {delta:.2f} pp > 0.5 pp noise floor, Teddy regression OK")
        print(f"Recommended: gf_radius={cfg3['gf_r']}, gf_eps={cfg3['gf_eps']:.0e}, wmf_radius={cfg3['wmf_r']}, lrc_threshold={cfg3['lrc']}")
    else:
        print(f"REJECT (gain too small or Teddy regression too big) → keep baseline {base}")


if __name__ == "__main__":
    main()

"""Hyperparameter sweep harness — v2 for AD-Census + sub-pixel pipeline.

Optimization target: minimise Codalab-style score = (Tsukuba_err × Venus_err × Teddy_err × time),
where time is the per-image processing time excluding the warm-up cost.

Calls computeDisp.py internals directly so we can patch hyperparameters without
mutating the production code. Reports per-config metrics + the leaderboard-shaped
product score.

Run:
    uv run python -m scripts.sweep
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

import computeDisp as cd

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"

CONFIG = {
    "Tsukuba": (15, 16, 8.0),   # (max_disp, scale_factor, threshold)
    "Venus":   (20, 8, 5.0),
    "Teddy":   (60, 4, 18.0),
    "Cones":   (60, 4, 15.0),
}
CODALAB_IMAGES = ["Tsukuba", "Venus", "Teddy"]   # the 3 Codalab images


def run_pipeline(
    Il: np.ndarray, Ir: np.ndarray, max_disp: int,
    gf_radius: int, gf_eps: float,
    wmf_radius: int,
    lrc_thresh: int,
    alpha: float, lam_c: float, lam_ad: float,
    subpixel: bool,
) -> np.ndarray:
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    census_L = cd.census_transform(gray_L)
    census_R = cd.census_transform(gray_R)
    c_census = cd.hamming_cost_volume(census_L, census_R, max_disp)
    if alpha > 0:
        c_ad = cd.ad_cost_volume(Il, Ir, max_disp)
        cost_L = cd.fuse_ad_census(c_census, c_ad, lam_c=lam_c, lam_ad=lam_ad, alpha=alpha)
    else:
        cost_L = c_census.astype(np.float32) / lam_c  # same normalization as fused
    cost_R = cd.reindex_cost_to_right(cost_L)
    cost_L_agg = cd.aggregate(cost_L, Il, radius=gf_radius, eps=gf_eps)
    cost_R_agg = cd.aggregate(cost_R, Ir, radius=gf_radius, eps=gf_eps)
    D_L = cd.winner_take_all(cost_L_agg)
    D_R = cd.winner_take_all(cost_R_agg)
    valid = cd.lr_consistency(D_L, D_R, threshold=lrc_thresh)
    filled = cd.hole_fill(D_L, valid, max_disp).astype(np.float32)
    if subpixel:
        d_sub = cd.subpixel_refine(D_L, cost_L_agg, max_disp)
        d_combined = np.where(valid, d_sub, filled)
        return cd.weighted_median_subpixel(d_combined, Il, max_disp, radius=wmf_radius).astype(np.uint8)
    else:
        disp_u8 = filled.astype(np.uint8)
        out = cv2.ximgproc.weightedMedianFilter(joint=Il, src=disp_u8, r=wmf_radius)
        return out.astype(np.uint8)


def evaluate_bpr(disp_pred: np.ndarray, disp_gt: np.ndarray, scale_factor: int, threshold: float = 1.0) -> float:
    """Vectorised replica of eval.py's evaluate()."""
    disp_p = np.int32(np.uint8(disp_pred * scale_factor) / scale_factor)
    disp_g = np.int32(disp_gt / scale_factor)
    mask = disp_g > 0
    if mask.sum() == 0:
        return 0.0
    errors = np.abs(disp_g[mask] - disp_p[mask]) > threshold
    return float(errors.sum()) / float(mask.sum())


def _load_dataset(name: str):
    Il = cv2.imread(str(TESTDATA / name / "img_left.png"))
    Ir = cv2.imread(str(TESTDATA / name / "img_right.png"))
    gt = cv2.imread(str(TESTDATA / name / "disp_gt.png"), -1)
    max_disp, scale, thresh = CONFIG[name]
    return Il, Ir, gt, max_disp, scale, thresh


def run_config(cfg: Dict, datasets: Iterable[str]) -> Dict[str, Dict[str, float]]:
    results = {}
    for name in datasets:
        Il, Ir, gt, max_disp, scale, thresh = _load_dataset(name)
        t0 = time.time()
        pred = run_pipeline(
            Il, Ir, max_disp,
            gf_radius=cfg["gf_r"], gf_eps=cfg["gf_eps"],
            wmf_radius=cfg["wmf_r"], lrc_thresh=cfg["lrc"],
            alpha=cfg["alpha"], lam_c=cfg["lam_c"], lam_ad=cfg["lam_ad"],
            subpixel=cfg["subpixel"],
        )
        dt = time.time() - t0
        bpr = evaluate_bpr(pred, gt, scale) * 100
        results[name] = {"bpr": bpr, "time": dt, "threshold": thresh, "margin": thresh - bpr}
    return results


def codalab_score(results: Dict[str, Dict[str, float]]) -> Tuple[float, float, bool]:
    """Returns (codalab_score, avg_bpr, all_pass)."""
    err_prod = 1.0
    time_sum = 0.0
    for name in CODALAB_IMAGES:
        r = results[name]
        err_prod *= r["bpr"]
        time_sum += r["time"]
    score = err_prod * time_sum
    all_bprs = [r["bpr"] for r in results.values()]
    avg_bpr = sum(all_bprs) / len(all_bprs)
    all_pass = all(r["bpr"] < r["threshold"] for r in results.values())
    return score, avg_bpr, all_pass


def fmt_row(name: str, r: Dict[str, float]) -> str:
    ok = "✓" if r["bpr"] < r["threshold"] else "✗"
    return (f"  {name:8s} BPR={r['bpr']:5.2f}% (th<{r['threshold']:4.1f}) "
            f"margin={r['margin']:+5.2f} t={r['time']:.2f}s {ok}")


def fmt_cfg(cfg: Dict) -> str:
    sub = "sub" if cfg["subpixel"] else "INT"
    return (f"α={cfg['alpha']:.1f} λc={cfg['lam_c']:.1f} λad={cfg['lam_ad']:.2f} "
            f"gf_r={cfg['gf_r']} gf_eps={cfg['gf_eps']:.0e} "
            f"wmf_r={cfg['wmf_r']} lrc={cfg['lrc']} {sub}")


def sweep(configs: List[Dict], datasets: Iterable[str]) -> List[Tuple[Dict, Dict, float, float, bool]]:
    rows = []
    for i, cfg in enumerate(configs):
        results = run_config(cfg, datasets)
        score, avg, ok = codalab_score(results)
        marker = "PASS" if ok else "FAIL"
        print(f"[{i+1:3d}/{len(configs)}] [{marker}] {fmt_cfg(cfg)}  "
              f"score={score:7.3f}  avg_bpr={avg:.2f}%")
        for ds, r in results.items():
            print(fmt_row(ds, r))
        rows.append((cfg, results, score, avg, ok))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="quick",
                        choices=["quick", "full", "alpha", "wmf", "tune_a0"],
                        help="quick: small grid; full: full grid; alpha/wmf: just one axis; tune_a0: gf×wmf with α=0")
    args = parser.parse_args()

    datasets = list(CONFIG.keys())

    # Baseline (current production)
    base = {"alpha": 0.5, "lam_c": 30.0, "lam_ad": 0.10,
            "gf_r": 7, "gf_eps": 1e-3, "wmf_r": 15, "lrc": 1, "subpixel": True}
    print("=" * 80)
    print("BASELINE (current production)")
    print("=" * 80)
    base_results = run_config(base, datasets)
    base_score, base_avg, _ = codalab_score(base_results)
    print(f"{fmt_cfg(base)}  score={base_score:.3f}  avg_bpr={base_avg:.2f}%")
    for ds, r in base_results.items():
        print(fmt_row(ds, r))

    # Build configs
    configs: List[Dict] = []
    if args.mode == "quick":
        # Critical question first: does AD-Census help at all?
        for alpha in [0.0, 0.3, 0.5, 1.0]:
            for sub in [True, False]:
                configs.append({**base, "alpha": alpha, "subpixel": sub})
    elif args.mode == "alpha":
        for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
            configs.append({**base, "alpha": alpha})
    elif args.mode == "wmf":
        for r in [9, 11, 13, 15, 17, 19, 21]:
            configs.append({**base, "wmf_r": r})
    elif args.mode == "full":
        alphas = [0.0, 0.3, 0.6, 1.0]
        lam_ads = [0.05, 0.10, 0.20]
        wmf_rs = [11, 13, 15, 17]
        for alpha, lam_ad, wmf_r in itertools.product(alphas, lam_ads, wmf_rs):
            configs.append({**base, "alpha": alpha, "lam_ad": lam_ad, "wmf_r": wmf_r})
    elif args.mode == "tune_a0":
        # alpha=0 confirmed best — sweep gf radius/eps + WMF radius around it
        a0 = {**base, "alpha": 0.0, "subpixel": True}
        for gf_r, gf_eps, wmf_r in itertools.product(
            [5, 7, 9, 11], [1e-4, 1e-3, 1e-2], [9, 11, 13, 15, 17]
        ):
            configs.append({**a0, "gf_r": gf_r, "gf_eps": gf_eps, "wmf_r": wmf_r})

    print(f"\n=== Sweep mode={args.mode}, {len(configs)} configs ===")
    rows = sweep(configs, datasets)

    # Final summary
    print("\n" + "=" * 80)
    print("Top 5 by codalab score (all-pass only)")
    print("=" * 80)
    passing = [(c, r, s, a) for c, r, s, a, ok in rows if ok]
    passing.sort(key=lambda x: x[2])
    for c, r, s, a in passing[:5]:
        print(f"  score={s:.3f}  avg_bpr={a:.2f}%  {fmt_cfg(c)}")

    if passing:
        best = passing[0]
        print(f"\n[BEST] score={best[2]:.3f} (vs baseline {base_score:.3f}) → "
              f"Δ {(base_score - best[2]) / base_score * 100:+.1f}%")
        print(f"  {fmt_cfg(best[0])}")


if __name__ == "__main__":
    main()

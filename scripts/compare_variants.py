"""Compare all stereo-matching variants on the 10-image local testset.

For each variant:
  - Tsukuba, Venus, Teddy, Cones (Codalab + threshold images)
  - Sawtooth, Bull, Barn1, Barn2, Poster, Map (6 generalisation images)

Reports per-image BPR + time, then the 3-image Codalab-style score
(Tsk_err × Vns_err × Tdy_err × sum_t) and the 6-gen average BPR. Picks the
variant with the lowest 3-image score that also satisfies the four BPR
thresholds and doesn't blow up generalisation BPR.

Run in the Codalab-equivalent venv for realistic timing:
    /tmp/codalab_env/bin/python -m scripts.compare_variants
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"

VARIANTS = [
    "v6_baseline",
    "v7_iter",
    "v7_crossscale",
    "v7_sgm",
    "v7_mst",
]

# (name, max_disp, scale_factor, threshold or None for generalisation set)
DATASETS = [
    ("Tsukuba", 15, 16, 8.0),
    ("Venus",   20, 8,  5.0),
    ("Teddy",   60, 4,  18.0),
    ("Cones",   60, 4,  15.0),
    ("Sawtooth", 20, 8, None),
    ("Bull",     18, 8, None),
    ("Barn1",    20, 8, None),
    ("Barn2",    17, 8, None),
    ("Poster",   21, 8, None),
    ("Map",      29, 8, None),
]

CODALAB_IMAGES = {"Tsukuba", "Venus", "Teddy"}


def evaluate_bpr(pred: np.ndarray, gt: np.ndarray, scale_factor: int) -> float:
    dp = np.int32(np.uint8(pred * scale_factor) / scale_factor)
    dg = np.int32(gt / scale_factor)
    mask = dg > 0
    if mask.sum() == 0:
        return 0.0
    return float((np.abs(dg[mask] - dp[mask]) > 1).mean() * 100)


def run_variant(name: str) -> Dict[str, Dict[str, float]]:
    mod = importlib.import_module(f"experiments.{name}")
    compute = mod.computeDisp

    results: Dict[str, Dict[str, float]] = {}
    # Warm cache + JITs.
    for ds_name, max_disp, _, _ in DATASETS:
        Il = cv2.imread(str(TESTDATA / ds_name / "img_left.png"))
        Ir = cv2.imread(str(TESTDATA / ds_name / "img_right.png"))
        if Il is not None and Ir is not None:
            try:
                _ = compute(Il, Ir, max_disp)
            except Exception:
                pass  # skip if a variant crashes on a particular image
            break  # one warm-up is enough

    for ds_name, max_disp, scale, threshold in DATASETS:
        Il = cv2.imread(str(TESTDATA / ds_name / "img_left.png"))
        Ir = cv2.imread(str(TESTDATA / ds_name / "img_right.png"))
        gt = cv2.imread(str(TESTDATA / ds_name / "disp_gt.png"), -1)
        if Il is None or Ir is None or gt is None:
            results[ds_name] = {"bpr": float("nan"), "time": float("nan"),
                                "threshold": threshold or 100.0, "missing": True}
            continue

        try:
            t0 = time.time()
            pred = compute(Il, Ir, max_disp)
            elapsed = time.time() - t0
            bpr = evaluate_bpr(pred, gt, scale)
        except Exception as exc:
            results[ds_name] = {"bpr": float("nan"), "time": float("nan"),
                                "threshold": threshold or 100.0, "error": str(exc)}
            continue

        results[ds_name] = {
            "bpr": bpr,
            "time": elapsed,
            "threshold": threshold or 100.0,
        }
    return results


def summarise(results: Dict[str, Dict[str, float]]) -> Tuple[float, float, float, bool, bool]:
    """Return (3-img score, 6-gen avg BPR, total 4-img time, 4-img pass, has_error)."""
    codalab_t = 0.0
    codalab_bpr_prod = 1.0
    fourimg_t = 0.0
    fourimg_pass = True
    gen_bprs: List[float] = []
    has_error = False
    for ds_name, max_disp, _, threshold in DATASETS:
        r = results.get(ds_name, {})
        if not r or np.isnan(r.get("bpr", float("nan"))):
            has_error = True
            continue
        if threshold is not None:  # one of 4 known
            fourimg_t += r["time"]
            if r["bpr"] >= r["threshold"]:
                fourimg_pass = False
            if ds_name in CODALAB_IMAGES:
                codalab_t += r["time"]
                codalab_bpr_prod *= r["bpr"]
        else:
            gen_bprs.append(r["bpr"])
    codalab_score = codalab_bpr_prod * codalab_t
    gen_avg = sum(gen_bprs) / len(gen_bprs) if gen_bprs else float("nan")
    return codalab_score, gen_avg, fourimg_t, fourimg_pass, has_error


def fmt_row(variant: str, results: Dict[str, Dict[str, float]]) -> str:
    cells = []
    for ds_name, _, _, _ in DATASETS:
        r = results.get(ds_name, {})
        if np.isnan(r.get("bpr", float("nan"))):
            cells.append("  ERR  ")
        else:
            cells.append(f"{r['bpr']:5.2f}/{r['time']:.2f}")
    return f"  {variant:18s}  " + "  ".join(cells)


def main() -> None:
    print(f"Running {len(VARIANTS)} variants on {len(DATASETS)} images in codalab-equivalent env.\n")

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in VARIANTS:
        print(f"[{variant}] running...")
        all_results[variant] = run_variant(variant)

    # Header.
    header = "  Variant".ljust(20) + " "
    for ds_name, _, _, _ in DATASETS:
        header += f"{ds_name[:8]:>9s}  "
    print()
    print("=" * (len(header) + 5))
    print("Per-image BPR(%) / time(s):")
    print(header)
    print("-" * (len(header) + 5))
    for variant in VARIANTS:
        print(fmt_row(variant, all_results[variant]))

    # Summary.
    print()
    print("=" * 80)
    print("Summary  (Codalab score = Tsk × Vns × Tdy × sum_t over the 3 graded images)")
    print("=" * 80)
    print(f"  {'Variant':18s}  {'3-img score':>11s}  {'4-img time':>10s}  {'6-gen avg':>9s}  {'pass4?':>6s}")
    rankings = []
    for variant in VARIANTS:
        r = all_results[variant]
        codalab_score, gen_avg, t4, pass4, err = summarise(r)
        flag = "✓" if pass4 else "✗"
        if err:
            flag = "ERR"
        print(f"  {variant:18s}  {codalab_score:>11.3f}  {t4:>10.3f}  {gen_avg:>9.3f}  {flag:>6s}")
        rankings.append((codalab_score, gen_avg, t4, pass4, err, variant))

    # Pick winner.
    print()
    candidates = [r for r in rankings if r[3] and not r[4]]  # pass4 AND no error
    if not candidates:
        print("No variant passed all 4 thresholds. Recommend keeping v6 baseline.")
        return
    # Filter by anti-overfit guard: 6-gen avg ≤ baseline + 0.5pp.
    baseline = next((r for r in rankings if r[5] == "v6_baseline"), None)
    if baseline is not None and not baseline[4]:
        base_gen = baseline[1]
        filtered = [r for r in candidates if r[1] <= base_gen + 0.5]
        if not filtered:
            print(f"All variants exceed anti-overfit guard ({base_gen:.2f} + 0.5pp on 6-gen).")
            filtered = candidates  # fall back to all passing
    else:
        filtered = candidates

    winner = min(filtered, key=lambda r: r[0])
    print(f"WINNER: {winner[5]}  (3-img score {winner[0]:.3f}, 6-gen avg {winner[1]:.2f})")


if __name__ == "__main__":
    main()

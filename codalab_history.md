# Codalab submission history — B11202015

Final_score on Codalab = `Tsukuba_err × Venus_err × Teddy_err × Total_time` (lower better).

The "local" column reports the equivalent product from our local `eval.py` run
on the 3 Codalab images (sum of per-image times). The ratio column highlights
how much time Codalab adds on top of our local numbers — useful for sanity-
checking whether time optimisations transferred.

| Date | Commit | Codalab Final | Codalab Tsk | Codalab Vns | Codalab Tdy | Codalab Time | Local product | Codalab time / local time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-05-26 | `de508b6` | **72.67** | 4.50 | 0.54 | 10.84 | 2.76 s | ~48 (1.84 s) | 1.50× |
| 2026-05-26 | `c7fdd04` | **50.30** | 3.81 | 0.41 | 9.72 | 3.34 s | ~14 (0.89 s) | 3.75× |
| 2026-05-26 | (pending v4) | _projected 14–22_ | 3.07 | 0.34 | 10.25 | _0.86 s codalab-env_ | ~9.2 (0.86 s) | — |
| 2026-05-27 | (pending v5) | _projected 8.5–14_ | 3.07 | 0.36 | 10.22 | _0.52 s codalab-env_ | ~5.9 (0.52 s) | — |

## Per-submission notes

### v1 — `de508b6`
- Pure Census + guided filter (radius=11, eps=1e-4) + integer WMF (radius=17)
- LUT-based popcount, single-direction Hamming reused for both L/R via full re-computation
- Placed last

### v5 — (pending upload)
- **Threading**: the per-disparity Hamming and guided-filter loops are now
  driven by `concurrent.futures.ThreadPoolExecutor` (workers = min(cpu_count,
  8)). numpy and cv2.ximgproc release the GIL during their C-level work, so
  multi-core graders see near-linear speedup up to ~4 workers.
- **R-aggregation tested and dropped (again)**: with threading making the
  second filter pass cheap, R-aggregation was easy to add back. Tested in
  codalab-env: brought Teddy/Cones down by ~1pp but pushed Tsukuba up from
  3.07 → 4.13 and Venus from 0.36 → 0.40, net BPR product worse. Kept off.
- Local-codalab-env 3-image time: 0.86 s → 0.52 s (-40%).
- BPR essentially unchanged from v4: Tsukuba 3.07 / Venus 0.36 / Teddy 10.22.
- Local-codalab-env 3-image score: 9.2 → **5.9**.
- Codalab projection with threading should narrow the local→Codalab ratio
  toward v1's 1.45× (vs v2/v3's likely 2.0–2.4×), since the extra Python-
  level overhead that hurt v2 is gone. Pessimistic projection: ~14.
  Optimistic projection: **8.5 — under 10**.
- Inspiration: a classmate's `computeDisp_Tasi.py` (score 15.38) revealed
  they used the same threading pattern. We weren't doing this; this is the
  most important single missing piece.

### v4 — `1008ddf`
- Verified Codalab environment via a fresh Python 3.8 + `pip install -r
  requirement.txt` venv: **numpy 1.24.4 + opencv 4.13** — confirms numpy lacks
  `bitwise_count`, so the SWAR fallback added in v3 is the real production
  path on the grader.
- Re-ran sweep with the grader-equivalent environment as the timing baseline.
  Top config: `gf_radius=7, gf_eps=1e-2, wmf_radius=15`.
- Dropped the [0, 255] disparity stretch around `weightedMedianFilter`.
  Sub-pixel float is cast directly to uint8 — saves a few numpy allocations
  per call and (surprisingly) reduces BPR. The implicit floor of negative
  parabolic deltas acts as a tie-breaker that helps the median.
- BPR vs v3 (codalab-env measurement): Tsukuba 3.21→3.07, Venus 0.34=0.34,
  Teddy 10.85→10.25, Cones 9.30→8.97.
- Local-codalab-env 3-image product: ~9.2 (vs v3's ~10.5). The simpler WMF
  call should also bring the Codalab time inflation closer to v1's 1.45×
  ratio (vs v2's 2.40× ratio) — the scaled WMF was the suspected cause.

### v3 — `e5c0bcf` (not uploaded; superseded by v4)
- SWAR popcount fallback replaces the byte-LUT path for numpy < 2.0
  (the suspected Codalab fallback). Local benchmark: 87 ms vs 301 ms for the
  Teddy Hamming pass — 3.5× faster than LUT, only 80 ms slower than
  `numpy.bitwise_count` (numpy ≥ 2.0).
- Skip aggregating the R-direction cost volume. The un-aggregated cost feeds
  the LR consistency mask directly. Outcome: Tsukuba/Venus BPR drops
  (un-aggregated D_R is more selective at occlusion edges); Teddy/Cones BPR
  rises modestly. 3-image BPR product drops 15.18 → 11.84.
- Re-swept gf radius / eps / wmf radius for the new pipeline:
  `gf_radius=5, gf_eps=3e-2, wmf_radius=15`.
- Net local: 3-image product score 14 → 8.9 (37% local improvement vs v2).
  Real Codalab score TBD; if the local→Codalab time ratio returns to ~1.5×
  thanks to SWAR (vs v2's 3.75×), we expect Codalab Final around 13–20.

### v2 — `c7fdd04`
- R→L cost via reindex (skip 2nd Hamming pass)
- `numpy.bitwise_count` popcount (15× faster than LUT, but only on numpy ≥ 2.0)
- guided filter eps relaxed: 1e-4 → 1e-2 (more smoothing)
- Parabolic sub-pixel refinement + WMF in scaled-up uint8 space
- AD-Census fusion implemented but evaluated as net-negative across 60 sweep
  configs; left disabled (`alpha=0`)
- Local BPR improved across all 4 images. Local time halved.
- **Codalab time INCREASED by 0.58 s** despite local time halving → Codalab
  almost certainly on numpy < 2.0 (no `bitwise_count`), so the LUT fallback
  runs, while the extra sub-pixel/scaled-WMF post-processing adds overhead the
  grader does see.

## Take-aways for future submissions

1. **Time on Codalab is the dominant lever now.** Codalab time / local time
   ratio jumped from 1.5× to 3.75× — the time factor in the score formula
   amplifies any slowdown on the grader machine.
2. **Don't assume numpy 2.x features transfer.** Anything that depends on
   `np.bitwise_count`, PEP 604 unions, etc. has a real chance of falling off
   the fast path on the grader.
3. **Local sweep score is an unreliable proxy for Codalab score.** Sub-pixel
   refinement improved local score by 0.6 units but apparently slowed the
   grader by ~0.6 s, which (multiplied by the BPR product) outweighs the gain.
4. **Heading**: try a v3 that strips sub-pixel + scaled WMF and goes back to
   integer WMF; measure the Codalab time delta. If time drops meaningfully,
   keep that path even at slightly worse local BPR.

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

## Per-submission notes

### v1 — `de508b6`
- Pure Census + guided filter (radius=11, eps=1e-4) + integer WMF (radius=17)
- LUT-based popcount, single-direction Hamming reused for both L/R via full re-computation
- Placed last

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

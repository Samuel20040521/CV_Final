# CV Final — Stereo Matching

NTU Computer Vision Spring 2026 final project. Two halves:

| Part | What | TA grading weight |
|---|---|---|
| **[Part 1 — Basic](#part-1--basic-middlebury-stereo)** | Disparity estimator scored on 4 Middlebury v2 pairs (Tsukuba / Venus / Teddy / Cones) via `eval.py` | 80% |
| **[Part 2 — Bonus](#part-2--bonus-tartanair-3d-reconstruction)** | Real-world stereo pipeline: TartanAir → v6 disparity → TSDF mesh → first-person walkthrough mp4 | 25% |

Branch `Teng` is where active bonus work lives; `main` carries the basic-task v6 baseline. Both halves are runnable from a fresh clone of `Teng`.

## Environment

```bash
uv sync                  # base deps (Part 1)
uv sync --extra bonus    # adds open3d, imageio, scipy (Part 2)
```

Python is pinned to 3.11 via `.python-version`. The grader runs Python 3.8+; the `computeDisp.py` implementation stays in 3.8-safe syntax.

---

## Part 1 — Basic (Middlebury stereo)

`computeDisp.py` is a flat single-file implementation of the standard 4-step Middlebury pipeline:

```
  Census transform → Hamming cost volume
  → Weighted Guided Image Filter aggregation (adaptive per-pixel eps)
  → Winner-take-all
  → LR consistency → hole fill → weighted median
```

It parallelises the per-disparity inner loops with a thread pool because `numpy` / `cv2.ximgproc` release the GIL.

### Run

```bash
uv run python eval.py --image Teddy        # target BPR <18%
uv run python eval.py --image Tsukuba      # target <8%
uv run python eval.py --image Venus        # target <5%
uv run python eval.py --image Cones        # target <15%
```

`main.py` is the same pipeline plus a PNG dump (`<name>.png`) for visual checking:

```bash
uv run python main.py --image Teddy
```

### Local baselines (Teng's machine)

| Image | BPR | Time |
|---|---|---|
| Teddy | 10.00% | 0.27 s |

Other three pairs only run on Codalab / the TA hidden image at submission time.

### TA-facing contract

`eval.py` and `main.py` both do `from computeDisp import computeDisp`, so `computeDisp.py` must remain at the repo root and expose that name. Do **not** edit `eval.py` (TA grading file). `main.py` CLI is `python3 main.py --image <name>` — don't change the interface.

### Layout

```
CV_Final/
├── eval.py                    grader file (do not edit)
├── main.py                    visualisation entry (do not edit interface)
├── computeDisp.py             flat v6 stereo implementation (~400 lines)
├── pyproject.toml             uv project + dependencies
├── requirement.txt            mirror of pyproject.toml for the TA's pip install
├── testdata/{Teddy,Tsukuba,Venus,Cones}/   Middlebury v2 pairs (committed)
└── experiments/eval_tartanair.py           BPR eval on TartanAir GT depth (Part 2 diagnostic)
```

---

## Part 2 — Bonus (TartanAir 3D reconstruction)

End-to-end pipeline: stereo matcher → TSDF fusion → first-person walkthrough mp4 of the reconstructed alley. Reuses `computeDisp` from Part 1 unchanged, so the bonus track can't affect the 80% score.

### TL;DR — three commands

```bash
# 1. Install bonus deps
uv sync --extra bonus

# 2. Fetch the demo dataset (~3 GB zip / ~11 GB unzipped)
uv run python -m bonus.download_tartanair

# 3. Render the walkthrough
uv run python -m bonus.run_bonus_tartanair --end 100 --out out/v6_demo.mp4
```

After step 3 you'll have an mp4 of v6's reconstruction of `japanesealley/Hard/P000` replayed from the original camera poses.

### Fetching the dataset

`bonus/download_tartanair.py` pulls from the official HuggingFace mirror (`theairlabcmu/tartanair`). The upstream AirLab Ceph host stalls badly for non-CMU networks (~25 KB/s with frequent SSL drops); HF sustains 5–10 MB/s and supports HTTP `Range` so resumes work cleanly.

```bash
# Default — just the demo dataset (japanesealley/Hard, ~3 GB)
uv run python -m bonus.download_tartanair

# Different scene/level
uv run python -m bonus.download_tartanair --scene office --level Easy

# Stereo only, no GT depth (skip the BPR eval part, saves ~1 GB)
uv run python -m bonus.download_tartanair --modalities image_left,image_right

# Every TartanAir V1 scene at the chosen level (large: 30-70 GB)
uv run python -m bonus.download_tartanair --all
uv run python -m bonus.download_tartanair --all --level Easy
```

The downloader uses `curl --retry-all-errors -C -` so partial files resume on the next run. Zips land in `data/tartanair_cache/` and are unzipped to `data/tartanair_raw/`.

After unzipping, one trajectory looks like:

```
data/tartanair_raw/japanesealley/Hard/P000/
├── image_left/  000XXX_left.png      640×480 PNG, rectified
├── image_right/ 000XXX_right.png     baseline 0.25 m
├── depth_left/  000XXX_left_depth.npy (480, 640) float32 metres
├── depth_right/ 000XXX_right_depth.npy
├── pose_left.txt                     tx ty tz qx qy qz qw per row, NED-body convention
└── pose_right.txt
```

Camera intrinsics are constant across TartanAir V1: `fx = fy = 320, cx = 320, cy = 240`.

### Running the 3D pipeline

```bash
# v6 prediction → TSDF → walkthrough (main demo)
uv run python -m bonus.run_bonus_tartanair --end 100 --out out/v6.mp4

# GT-depth reference → TSDF → walkthrough (pipeline upper bound)
uv run python -m bonus.run_bonus_tartanair --use-gt-depth --end 100 --out out/gt.mp4

# Different trajectory or longer cut
uv run python -m bonus.run_bonus_tartanair --traj P001 --end 200 --out out/p001.mp4
```

Pipeline stages (`bonus/run_bonus_tartanair.py`):

1. `bonus.tartanair.list_frames` / `load_poses` — read TartanAir files; convert pose from NED body axes (x=fwd, y=right, z=down) into OpenCV camera convention (x=right, y=down, z=fwd) via a right-multiplied axis permutation. Without this conversion, depth gets back-projected along world-z (vertical) instead of along the alley.
2. `computeDisp(Il, Ir, max_disp)` — v6 disparity, OR `np.load(depth_left.npy)` if `--use-gt-depth`.
3. `bonus.depth.disparity_to_depth(disp, fx, baseline)` — Z = fx·B / d.
4. Mask sky (depth ≥ 65000, which is float16 max — TartanAir's inf marker) and far-range pixels (depth > `--depth-trunc`).
5. `bonus.fusion.integrate_frame` — Open3D ScalableTSDFVolume integration.
6. `bonus.render.render_mesh_follow_trajectory` — re-render the fused mesh from each original capture pose. This gives a video that overlays the reconstruction with the input sequence and sidesteps the "where's up in this world frame" question that orbit rendering would need to solve.

### BPR diagnostic

Optional — run v6 on TartanAir frames and compute Middlebury-style BPR against GT depth (converted to GT disparity via fx·B / depth):

```bash
uv run python -m experiments.eval_tartanair --end 100               # default: Hard P000, threshold 1px
uv run python -m experiments.eval_tartanair --threshold 3.0         # KITTI / TartanAir convention
uv run python -m experiments.eval_tartanair --traj P001 --max-disp 192   # try other trajectories
```

Baseline result on `japanesealley/Hard/P000` frames 0..99 with `max_disp=64`:

| Threshold | mean BPR | median | worst | best |
|---|---|---|---|---|
| 1.0 px | 20.21% | 20.17% | 33.31% | 9.51% |
| 3.0 px | 10.75% | 10.18% | 26.86% | 2.83% |

Time: 0.36 s/frame. We tried other scenes/trajectories looking for a "better" baseline — Easy levels and indoor `office/Easy` were *worse* (40–46% at 1 px) because they have closer objects (wider disparity range → wider search → more ambiguity) or texture-less surfaces. P000's far-distance trajectory is unusually well-suited to classical stereo.

The bonus demo doesn't depend on hitting a specific BPR — the 20% number is just an honest report of what v6 can do on this dataset, comparable to classical stereo numbers in the TartanAir literature.

### Bonus layout

```
bonus/
├── download_tartanair.py       ★ HF-mirror downloader; --all fetches every V1 scene
├── tartanair.py                ★ V1 loader: intrinsics + frames + pose (NED → OpenCV)
├── run_bonus_tartanair.py      ★ active CLI: stereo → TSDF → first-person walkthrough mp4
├── depth.py                    disparity → metric depth
├── fusion.py                   Open3D ScalableTSDFVolume integration + mesh extract
├── render.py                   render_mesh_follow_trajectory + orbit fallback
├── single_frame.py             single-frame disparity + point-cloud preview (KITTI)
├── kitti.py                    legacy — KITTI raw loader (OXTS, calib_cam_to_cam)
├── download_sample.py          legacy — fetch a small KITTI raw drive
├── rectify.py                  legacy — from-scratch stereo rectification (for KITTI raw)
└── run_bonus.py                legacy — KITTI CLI

experiments/
└── eval_tartanair.py           Middlebury-style BPR on TartanAir GT depth

data/                           ★ gitignored; populated by download scripts
├── tartanair_raw/<scene>/<level>/P00X/...
└── tartanair_cache/                       (zip cache, kept for resume)
```

The KITTI files are kept as reference / report comparison — they were the original bonus track before we switched to TartanAir to get ground-truth poses + GT depth in one package.

---

## Submission contract recap

- 6/1 23:59 — code freeze, email to `fushengyu@media.ee.ntu.edu.tw`. Contents: Python code + a `.txt` with current local numbers. TA runs the hidden image with our 6/1 code on 6/2.
- 6/5 23:59 — report (PDF + slides) + bonus code + a `.txt` explaining bonus CLI, to the same email.
- Don't edit `eval.py`. Don't hard-code per-image hyperparameters (TA's 2026-05-27 invariance rule). `Cones` won't be the hidden image.

See `CLAUDE.md` for the full set of constraints.

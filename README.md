# CV Final — Stereo Matching

NTU Computer Vision Spring 2026 final project. Implements the 4-step Middlebury stereo matching pipeline (Census cost → guided-filter aggregation → winner-take-all → LR check + hole fill + weighted median).

## Quick start (uv)

```bash
uv sync
uv run python eval.py --image Teddy
```

The first command creates `.venv/` from `pyproject.toml` / `uv.lock` and pins Python 3.11.
The second runs the full pipeline and prints the bad-pixel ratio.

Run on any of the four Middlebury v2 pairs (drop them into `testdata/<name>/`):

```bash
uv run python eval.py --image Tsukuba   # target <8%
uv run python eval.py --image Venus     # target <5%
uv run python eval.py --image Teddy     # target <18%
uv run python eval.py --image Cones     # target <15%
```

`main.py` is equivalent but also writes `<name>.png` (the scaled disparity map) to the project root.

## Layout

```
CV_Final/
├── pyproject.toml          uv project + dependencies
├── .python-version         pins Python 3.11
├── eval.py                 grading script (do not edit)
├── main.py                 entry point with visualization
├── computeDisp.py          shim: re-exports computeDisp from the package
├── stereo_matching/        the implementation
│   ├── pipeline.py         orchestrator — public computeDisp()
│   ├── cost.py             Step 1: census transform + Hamming cost volume
│   ├── aggregation.py      Step 2: guided-filter aggregation
│   ├── optimization.py     Step 3: winner-take-all
│   └── refinement.py       Step 4: LR consistency → hole fill → weighted median
├── testdata/               Middlebury v2 pairs (img_left.png, img_right.png, disp_gt.png)
└── requirement.txt         grader-facing dependency list (mirror of pyproject.toml)
```

## Why this layout

`eval.py` is uneditable and does `from computeDisp import computeDisp`, so a top-level
`computeDisp.py` must remain. It is a one-line shim that re-exports the implementation
from the `stereo_matching` package. This keeps the assignment's import contract intact
while allowing the algorithm to live in proper submodules.

The package is flat (no `src/`) so `python eval.py` works without an install step.

## Bonus track — real-world / photoreal stereo

The `bonus/` package builds an end-to-end depth pipeline on stereo footage and
produces a flythrough video of the reconstructed scene. It reuses
`stereo_matching.computeDisp` unchanged, so the bonus pipeline cannot affect
the main 80% score.

Install bonus deps (open3d, imageio, scipy):

```bash
uv sync --extra bonus
```

### Active dataset — TartanAir V1 (default: `japanesealley/Hard`)

TartanAir gives us **stereo pairs + GT depth + GT camera poses** in one
package, so we can validate both disparity quality (BPR vs. GT depth) and
trajectory-conditioned 3D fusion. Source: the official Hugging Face mirror
`theairlabcmu/tartanair`.

```bash
uv run python -m bonus.download_tartanair
```

That fetches `japanesealley/Hard` — six trajectories `P000..P005` totalling
2705 stereo frames. Wire size: ~3 GB of zip cache, ~11 GB after unzip. The
script is resumable; re-running it skips files that are already complete.

Common variants:

```bash
# Different scene/level
uv run python -m bonus.download_tartanair --scene office --level Easy

# Stereo only, no GT depth (saves ~1 GB)
uv run python -m bonus.download_tartanair --modalities image_left,image_right

# Keep zips, don't unzip
uv run python -m bonus.download_tartanair --no-unzip
```

After download, each `P00X/` contains:

| Item | Format | Notes |
|---|---|---|
| `image_left/000XXX_left.png` | 640×480 PNG | Already rectified, pinhole |
| `image_right/000XXX_right.png` | 640×480 PNG | Baseline 0.25 m |
| `depth_left/000XXX_left_depth.npy` | (480,640) float32 | Metres; sky/inf clipped to 65504 |
| `depth_right/000XXX_right_depth.npy` | (480,640) float32 | |
| `pose_left.txt` | one row per frame | `tx ty tz qx qy qz qw`, unit quaternion |
| `pose_right.txt` | one row per frame | Same format |

Camera intrinsics (constant): `fx = fy = 320, cx = 320, cy = 240`.

**Why the HuggingFace mirror, not the upstream AirLab Ceph host?** Two reasons.
First, the AirLab endpoint `airlab-cloud.andrew.cmu.edu` stalls badly for
non-CMU networks: we measured ~25 KB/s averaged across mid-transfer SSL
timeouts. The HF CDN sustains 5-10 MB/s here. Second, HF supports `Range`
requests so the downloader resumes partial zips cleanly. The downloader's
`curl` flags (`--retry-all-errors`, `--speed-limit/--speed-time`) are tuned
for the failure modes we saw on AirLab; they don't hurt on HF.

### Legacy — KITTI raw

The original bonus track used KITTI raw drives (real driving footage, GPS/IMU
poses, no GT depth). Code under `bonus/kitti.py`, `bonus/download_sample.py`,
and `bonus/run_bonus.py` is preserved for report comparison.

```bash
uv run python -m bonus.download_sample --root data/kitti_raw  # ~60 MB
uv run python -m bonus.run_bonus \
    --kitti-root data/kitti_raw \
    --date 2011_09_26 --drive 0005 \
    --start 0 --end 100 --stride 2 \
    --max-disp 96 \
    --rectify scratch \
    --out out/bonus_demo.mp4
```

`--rectify scratch` uses the from-scratch rectification (textbook decomposition
of R, T into rectification homographies, then `cv2.remap`). `--rectify kitti`
uses the rectification rotations KITTI publishes — switch between them for
the report.

### Bonus layout

```
bonus/
├── download_tartanair.py  ★ active — fetch a TartanAir V1 sequence (HF mirror)
├── download_sample.py        legacy — fetch a small KITTI raw drive
├── kitti.py                  KITTI raw loader (calibration + OXTS poses)
├── rectify.py                calibrated stereo rectification from first principles
├── depth.py                  disparity → metric depth → colored point cloud
├── fusion.py                 Open3D ScalableTSDFVolume multi-frame integration
├── render.py                 offscreen orbit flythrough → mp4
├── single_frame.py           one-shot disparity + point-cloud preview
└── run_bonus.py              end-to-end CLI (KITTI)
```

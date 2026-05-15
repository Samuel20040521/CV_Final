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

## Bonus track — real-world stereo on KITTI raw

The `bonus/` package builds an end-to-end depth pipeline on real driving footage
and produces a flythrough video of the reconstructed scene. It reuses
`stereo_matching.computeDisp` unchanged, so it cannot affect the main 80% score.

Install bonus deps (open3d, imageio, scipy):

```bash
uv sync --extra bonus
```

Download a small KITTI raw drive (~60 MB):

```bash
uv run python -m bonus.download_sample --root data/kitti_raw
```

Run the full pipeline on the first 100 frames at stride 2:

```bash
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
uses the rectification rotations KITTI publishes, which is the ground truth and
the more practical choice for the demo — switch between them for the report.

Bonus layout:

```
bonus/
├── kitti.py            KITTI raw loader (calibration + OXTS poses)
├── rectify.py          calibrated stereo rectification from first principles
├── depth.py            disparity → metric depth → colored point cloud
├── fusion.py           Open3D ScalableTSDFVolume multi-frame integration
├── render.py           offscreen orbit flythrough → mp4 (imageio + ffmpeg)
├── run_bonus.py        end-to-end CLI
└── download_sample.py  fetch a small KITTI raw drive for testing
```

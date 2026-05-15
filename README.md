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

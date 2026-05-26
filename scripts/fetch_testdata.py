"""Fetch Tsukuba/Venus/Cones from Middlebury public archives into testdata/.

This script is **not** part of the submission — it lives in scripts/ for local
validation only. The course officially only ships Teddy in testdata/; the
remaining 3 images come from the same Middlebury archives the assignment is
based on.

For each dataset, normalises filenames + format to match `testdata/Teddy/`:
- img_left.png  (BGR uint8)
- img_right.png (BGR uint8)
- disp_gt.png   (uint8, value = true_disparity * scale_factor)

scale_factor values come from eval.py and follow the Middlebury convention:
- Tsukuba: 16 (scenes2001)
- Venus:    8 (scenes2001)
- Cones:    4 (scenes2003)
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "middlebury_raw"
TESTDATA = ROOT / "testdata"

SOURCES = {
    "Tsukuba": {
        "url": "https://vision.middlebury.edu/stereo/data/scenes2001/data/tsukuba/tsukuba.zip",
        "left_inside_zip": "scene1.row3.col3.ppm",
        "right_inside_zip": "scene1.row3.col4.ppm",
        "gt_inside_zip": "truedisp.row3.col3.pgm",
        "expected_scale": 16,
    },
    "Venus": {
        "url": "https://vision.middlebury.edu/stereo/data/scenes2001/data/venus/venus.zip",
        "left_inside_zip": "im2.ppm",
        "right_inside_zip": "im6.ppm",
        "gt_inside_zip": "disp2.pgm",
        "expected_scale": 8,
    },
    "Cones": {
        "url": "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/cones-png-2.zip",
        "left_inside_zip": "im2.png",
        "right_inside_zip": "im6.png",
        "gt_inside_zip": "disp2.png",
        "expected_scale": 4,
    },
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] already downloaded: {dest.name}")
        return
    print(f"[fetch] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"[fetch] saved {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")


def _extract_one(zip_path: Path, member: str, out_dir: Path) -> Path:
    """Extract a single file from the zip — handles flat or nested layouts."""
    with zipfile.ZipFile(zip_path) as zf:
        # Find a name that ENDS WITH the requested member (handles subdir layouts)
        matches = [n for n in zf.namelist() if n.endswith(member)]
        if not matches:
            raise FileNotFoundError(f"{member} not in {zip_path.name}; have: {zf.namelist()}")
        chosen = matches[0]
        out_path = out_dir / Path(chosen).name
        with zf.open(chosen) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def _convert_to_png(src: Path, dst: Path) -> None:
    """Read with cv2 (handles ppm/pgm/png), write as PNG. Preserves dtype."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cv2 failed to read {src}")
    cv2.imwrite(str(dst), img)


def fetch_one(name: str) -> None:
    cfg = SOURCES[name]
    dest_dir = TESTDATA / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    zip_path = RAW / Path(cfg["url"]).name
    _download(cfg["url"], zip_path)

    # 2. Extract the 3 files we need into a scratch dir
    scratch = RAW / f"_extract_{name}"
    scratch.mkdir(exist_ok=True)
    left_src = _extract_one(zip_path, cfg["left_inside_zip"], scratch)
    right_src = _extract_one(zip_path, cfg["right_inside_zip"], scratch)
    gt_src = _extract_one(zip_path, cfg["gt_inside_zip"], scratch)

    # 3. Convert to the testdata/Teddy/-style PNG naming
    _convert_to_png(left_src, dest_dir / "img_left.png")
    _convert_to_png(right_src, dest_dir / "img_right.png")
    _convert_to_png(gt_src, dest_dir / "disp_gt.png")

    # 4. Verify the GT scale matches eval.py expectations
    gt = cv2.imread(str(dest_dir / "disp_gt.png"), -1)
    max_val = int(gt.max())
    inferred_disp = max_val / cfg["expected_scale"]
    print(
        f"[{name}] disp_gt: shape={gt.shape} dtype={gt.dtype} "
        f"max={max_val} → true_disp ≈ {inferred_disp:.1f} "
        f"(scale_factor={cfg['expected_scale']})"
    )


def main() -> None:
    for name in SOURCES:
        fetch_one(name)
    print("\n[done] testdata/ now contains:")
    for d in sorted(TESTDATA.iterdir()):
        if d.is_dir():
            files = sorted(p.name for p in d.iterdir())
            print(f"  {d.name}/ → {files}")


if __name__ == "__main__":
    main()

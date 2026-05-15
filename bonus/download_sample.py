"""Fetch a small KITTI raw drive (calibration + a short sync drive).

Defaults to 2011_09_26 calib + drive 0005 (~150 frames, ~60 MB). The
synced+rectified drive uses image_02 (color left) and image_03 (color right);
we load those plus the OXTS records.

Usage:
    uv run python -m bonus.download_sample --root data/kitti_raw

After this completes, run:
    uv run python -m bonus.run_bonus --kitti-root data/kitti_raw \
        --date 2011_09_26 --drive 0005 --end 100 --stride 2 \
        --out out/bonus_demo.mp4
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"
DRIVE_URL_TPL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "2011_09_26_drive_{drive}/2011_09_26_drive_{drive}_sync.zip"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] already downloaded: {dest.name}")
        return
    print(f"[get] {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fp:
        shutil.copyfileobj(resp, fp)


def _unzip(zip_path: Path, target: Path) -> None:
    print(f"[unzip] {zip_path.name} → {target}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=Path("data/kitti_raw"), type=Path,
                   help="where to put the KITTI tree")
    p.add_argument("--drive", default="0005",
                   help="drive id (4 digits) under 2011_09_26. Defaults to 0005.")
    args = p.parse_args()

    cache = args.root.parent / "kitti_cache"
    cache.mkdir(parents=True, exist_ok=True)

    calib_zip = cache / "2011_09_26_calib.zip"
    drive_zip = cache / f"2011_09_26_drive_{args.drive}_sync.zip"

    _download(CALIB_URL, calib_zip)
    _download(DRIVE_URL_TPL.format(drive=args.drive), drive_zip)

    args.root.mkdir(parents=True, exist_ok=True)
    _unzip(calib_zip, args.root)
    _unzip(drive_zip, args.root)
    print(f"[done] KITTI sample ready at {args.root}/2011_09_26/")


if __name__ == "__main__":
    main()

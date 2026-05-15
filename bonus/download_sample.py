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
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"
DRIVE_URL_TPL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "2011_09_26_drive_{drive}/2011_09_26_drive_{drive}_sync.zip"
)


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with a tqdm progress bar.

    KITTI's S3 mirror is slow (often <200 KB/s), so silent downloads look stuck.
    Resumes by checking the existing local size against Content-Length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Verify completeness via HEAD; if it matches Content-Length, skip.
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as r:
                expected = int(r.headers.get("Content-Length", "0"))
            if expected and dest.stat().st_size == expected:
                print(f"[skip] already downloaded: {dest.name} ({expected} bytes)")
                return
            print(f"[resume] {dest.name}: have {dest.stat().st_size}/{expected} bytes")
        except Exception as e:
            print(f"[warn] HEAD check failed ({e}); re-downloading {dest.name}")

    print(f"[get] {url}")
    headers = {}
    mode = "wb"
    start = 0
    if dest.exists():
        start = dest.stat().st_size
        headers["Range"] = f"bytes={start}-"
        mode = "ab"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0")) + start
        chunk = 64 * 1024
        with open(dest, mode) as fp, tqdm(
            total=total, initial=start, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fp.write(buf)
                bar.update(len(buf))


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

"""Fetch one TartanAir V1 sequence from the Hugging Face mirror.

Defaults to ``japanesealley/Hard``, the smallest stereo sequence that also ships
ground-truth depth + ground-truth poses. About 3 GB downloaded (zip cache),
~11 GB after unzip.

Why HuggingFace instead of the upstream AirLab Ceph endpoint:
the AirLab host ``airlab-cloud.andrew.cmu.edu`` stalls for users outside the
CMU network (we measured ~25 KB/s with frequent mid-transfer SSL timeouts),
while the HuggingFace CDN sustains 5-10 MB/s and supports ``Range`` requests
so partial files resume cleanly.

Usage::

    uv run python -m bonus.download_tartanair                       # defaults
    uv run python -m bonus.download_tartanair --scene office --level Easy
    uv run python -m bonus.download_tartanair --modalities image_left,image_right  # stereo only

Final layout (default scene/level)::

    data/tartanair_raw/japanesealley/Hard/
    ├── P000/  P001/ ... P005/
    │   ├── image_left/000XXX_left.png        640x480 PNG, rectified
    │   ├── image_right/000XXX_right.png      baseline = 0.25 m
    │   ├── depth_left/000XXX_left_depth.npy  (480,640) float32 metres
    │   ├── depth_right/000XXX_right_depth.npy
    │   ├── pose_left.txt                     7 cols: tx ty tz qx qy qz qw
    │   └── pose_right.txt

Camera intrinsics (all frames): ``fx = fy = 320, cx = 320, cy = 240``.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


HF_BASE = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main"
MODALITIES = ("image_left", "image_right", "depth_left", "depth_right")


def _curl_resume(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with retry + resume.

    The flags here matter; we hit two real failure modes during development:

    * ``--retry-all-errors`` — curl's default ``--retry`` only handles
      connection-establishment errors. Mid-transfer SSL_read timeouts
      (exit 56) are *not* covered without this flag.
    * ``--speed-limit 100KB/s --speed-time 60s`` — aborts a stalled
      transfer fast enough that the outer retry loop can reopen the
      connection, instead of sitting at 0 B/s for half an hour.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fL",
        "--retry-all-errors", "--retry", "10", "--retry-delay", "5",
        "--connect-timeout", "30",
        "--speed-limit", "102400", "--speed-time", "60",
        "-C", "-",
        "-o", str(dest), url,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit("[error] curl exit {} for {}".format(rc, url))


def _unzip(zip_path: Path, target: Path) -> None:
    """Extract ``zip_path`` into ``target``.

    ``-o`` overwrites silently; pose_*.txt is duplicated across the four
    modality zips with identical content, so overwrite is harmless.
    """
    target.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(target)]
    ).returncode
    # 0 = OK; 1 = warnings; 2 = directory mkdir conflicts (benign with -o
    # when multiple zips share a tree).
    if rc not in (0, 1, 2):
        sys.exit("[error] unzip exit {} for {}".format(rc, zip_path))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download one TartanAir V1 sequence from the HuggingFace mirror."
    )
    p.add_argument("--scene", default="japanesealley",
                   help="scene name (lowercase, e.g. japanesealley, office, carwelding)")
    p.add_argument("--level", default="Hard", choices=["Easy", "Hard"],
                   help="trajectory difficulty (Easy or Hard)")
    p.add_argument("--modalities", default=",".join(MODALITIES),
                   help="comma-separated subset of {}".format(MODALITIES))
    p.add_argument("--cache", default=Path("data/tartanair_cache"), type=Path,
                   help="zip download dir (kept after unzip so re-runs resume fast)")
    p.add_argument("--root", default=Path("data/tartanair_raw"), type=Path,
                   help="unzip destination root")
    p.add_argument("--no-unzip", action="store_true",
                   help="download zips only; skip extraction")
    args = p.parse_args()

    if shutil.which("curl") is None:
        sys.exit("[error] curl not found in PATH (required for HuggingFace download)")
    if not args.no_unzip and shutil.which("unzip") is None:
        sys.exit("[error] unzip not found in PATH (required to extract; pass --no-unzip to skip)")

    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    unknown = [m for m in mods if m not in MODALITIES]
    if unknown:
        sys.exit("[error] unknown modalities: {} (valid: {})".format(unknown, MODALITIES))

    args.cache.mkdir(parents=True, exist_ok=True)
    zips: List[Path] = []
    for mod in mods:
        url = "{}/{}/{}/{}.zip".format(HF_BASE, args.scene, args.level, mod)
        dest = args.cache / "{}_{}_{}.zip".format(args.scene, args.level, mod)
        print("[get] {}".format(url))
        _curl_resume(url, dest)
        zips.append(dest)

    if args.no_unzip:
        print("[done] cached {} zip(s) at {}".format(len(zips), args.cache))
        return

    args.root.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print("[unzip] {} -> {}".format(z.name, args.root))
        _unzip(z, args.root)
    print("[done] TartanAir {}/{} ready at {}/{}/{}".format(
        args.scene, args.level, args.root, args.scene, args.level))


if __name__ == "__main__":
    main()

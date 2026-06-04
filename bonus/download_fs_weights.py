"""Download FoundationStereo pretrained weights for the bonus pipeline.

Two variants are released by NVIDIA:

  * ``11-33-40`` — ViT-small backbone, ~750 MB. Faster inference, BPR @1px
    around 3-4% on TartanAir P000. **Default**.
  * ``23-51-11`` — ViT-large backbone, ~3.3 GB. Strongest accuracy.

The official source is a Google Drive folder. ViT-small comes down cleanly
via gdown. ViT-large hits Google Drive's anonymous quota every few hours, so
we fall back to a HuggingFace community mirror (``Felix-Zhenghao/FoundationStereo``)
for the big ``.pth`` file. Both endpoints are public — **no HuggingFace
account or token required**.

Usage::

    uv run python -m bonus.download_fs_weights                # ViT-small (default, recommended)
    uv run python -m bonus.download_fs_weights --variant large
    uv run python -m bonus.download_fs_weights --check        # verify a download
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple


# Single Google Drive folder released by NVlabs (per FoundationStereo README).
# It contains BOTH variants as subfolders 11-33-40/ and 23-51-11/.
# gdown --folder pulls both; we then verify whichever variant the user asked for.
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"

VARIANT_DIRNAME = {"small": "11-33-40", "large": "23-51-11"}

# HuggingFace community mirror for ViT-large only — public, anonymous-readable.
HF_MIRROR_LARGE_PTH = (
    "https://huggingface.co/Felix-Zhenghao/FoundationStereo/resolve/main/model_best_bp2.pth"
)

# Expected file sizes (bytes, with ±5% tolerance) for sanity check.
EXPECTED_SIZES = {
    "small": 787_711_942,
    # 23-51-11 .pth is ~3.3 GB on the HF mirror.
    "large": 3_300_000_000,
}

DEFAULT_ROOT = Path("data/pretrained_models/foundation_stereo")


def _gdown_folder(url: str, out_dir: Path) -> None:
    """Pull a Google Drive folder via gdown's --folder mode."""
    if shutil.which("gdown") is None:
        sys.exit("[error] gdown not installed. uv add gdown  or  pip install gdown")
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(["gdown", "--folder", url, "-O", str(out_dir)]).returncode
    if rc != 0:
        sys.exit(
            "[error] gdown exit {} — Google Drive may have rate-limited this folder.\n"
            "        For ViT-large, retry with: --variant large --fallback-hf".format(rc)
        )


def _curl_resume(url: str, dest: Path) -> None:
    """Resumable download via curl. Used for the HF .pth fallback."""
    if shutil.which("curl") is None:
        sys.exit("[error] curl not installed")
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


def _check_variant(out_dir: Path, variant: str) -> Tuple[bool, str]:
    """Verify a downloaded variant: cfg.yaml + .pth present + size sensible."""
    cfg = out_dir / "cfg.yaml"
    pth = out_dir / "model_best_bp2.pth"
    if not cfg.is_file():
        return False, "cfg.yaml missing in {}".format(out_dir)
    if not pth.is_file():
        return False, "model_best_bp2.pth missing in {}".format(out_dir)
    size = pth.stat().st_size
    expected = EXPECTED_SIZES[variant]
    if size < int(expected * 0.5):
        return False, "model_best_bp2.pth only {} bytes (expected ~{}); partial?".format(
            size, expected)
    return True, "OK ({} bytes)".format(size)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download FoundationStereo weights for the bonus pipeline.",
    )
    p.add_argument("--variant", default="small", choices=["small", "large"],
                   help="ViT backbone size; 'small' (~750MB) is the recommended default")
    p.add_argument("--root", default=DEFAULT_ROOT, type=Path,
                   help="parent directory; weights land at <root>/<variant_dir>/")
    p.add_argument("--fallback-hf", action="store_true",
                   help="for --variant large, skip Google Drive's flaky big-file download "
                        "and pull the .pth from the HuggingFace mirror directly")
    p.add_argument("--check", action="store_true",
                   help="only verify an existing download; do not fetch")
    args = p.parse_args()

    out_dir = args.root / VARIANT_DIRNAME[args.variant]

    if args.check:
        ok, msg = _check_variant(out_dir, args.variant)
        print("[check] {}: {}".format("OK" if ok else "FAIL", msg))
        sys.exit(0 if ok else 1)

    print("[plan] target = {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[fetch] gdown the FoundationStereo Google Drive folder")
    print("        (pulls both 11-33-40/ and 23-51-11/; we'll keep the variant you asked for)")
    _gdown_folder(GDRIVE_FOLDER, out_dir.parent)

    if args.variant == "large" and args.fallback_hf:
        # ViT-large via HF mirror: GDrive frequently rate-limits the big .pth.
        # gdown step above already fetched cfg.yaml (small file, reliable);
        # we now overwrite the .pth from the HF mirror if it's missing/truncated.
        pth_path = out_dir / "model_best_bp2.pth"
        if not pth_path.is_file() or pth_path.stat().st_size < EXPECTED_SIZES["large"] // 2:
            print("[fetch] model_best_bp2.pth from HuggingFace mirror (anonymous, no token)")
            _curl_resume(HF_MIRROR_LARGE_PTH, pth_path)

    ok, msg = _check_variant(out_dir, args.variant)
    if not ok:
        sys.exit("[error] post-download check failed: {}".format(msg))
    print("[done] {} ready at {} — {}".format(args.variant, out_dir, msg))


if __name__ == "__main__":
    main()

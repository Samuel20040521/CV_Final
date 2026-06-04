"""Deep stereo wrapper for the bonus pipeline.

Currently supports **FoundationStereo** (Wen, Trepte et al. CVPR 2025 Oral —
Best Paper Nomination). The checkpoint and source live outside the repo:

  - source: ``third_party/FoundationStereo/`` (git clone of NVlabs/FoundationStereo)
  - weights: ``data/pretrained_models/foundation_stereo/<run>/`` containing
    ``cfg.yaml`` + ``model_best_bp2.pth`` exactly as released on Google Drive

Why this module instead of extending ``computeDisp.py``: the basic-task
``computeDisp`` is TA-graded and must stay untouched. Keeping the deep matcher
in a separate module also lets us return ``float32`` sub-pixel disparity
instead of clipping to ``uint8`` — the v8 experiment proved sub-pixel matters
for the downstream TSDF integration.

Public entry point::

    compute_disparity_deep(Il_bgr, Ir_bgr, ckpt_dir) -> np.float32 (H, W)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# Path to the vendored FoundationStereo source. We inject it into ``sys.path``
# lazily inside ``_load_model`` so that ``import bonus.deep_matcher`` never
# triggers a heavy torch import at module-load time.
FOUNDATION_STEREO_REPO = Path(__file__).resolve().parent.parent / "third_party" / "FoundationStereo"


# Singleton cache so the second call doesn't re-build + re-load 1.5 GB of
# weights. Keyed by ``str(ckpt_dir)`` because Path equality is fine but plain
# strings serialise more predictably in a logging context.
_MODEL_CACHE: dict = {}


def _ensure_repo_on_path() -> None:
    """Add ``third_party/FoundationStereo`` to ``sys.path`` exactly once.

    The FoundationStereo code uses absolute imports like ``from core.foundation_stereo
    import FoundationStereo`` and ``from Utils import ...``, so the repo root
    has to be importable from the top level.
    """
    repo_str = str(FOUNDATION_STEREO_REPO)
    if not FOUNDATION_STEREO_REPO.is_dir():
        raise FileNotFoundError(
            "FoundationStereo source not found at {}. "
            "Run: git clone https://github.com/NVlabs/FoundationStereo.git {}".format(
                FOUNDATION_STEREO_REPO, FOUNDATION_STEREO_REPO))
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_model(ckpt_dir: Path, device: str = "cuda"):
    """Build FoundationStereo from ``cfg.yaml`` + load weights into eval mode.

    ``ckpt_dir`` may be either the directory containing ``cfg.yaml`` +
    ``model_best_bp2.pth`` *or* the .pth path itself (the demo accepts both;
    we accept both for symmetry). The cached model is keyed by the .pth path.
    """
    import torch  # local import — keeps module import cheap
    from omegaconf import OmegaConf

    ckpt_path = Path(ckpt_dir)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "model_best_bp2.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError("FoundationStereo checkpoint not found: {}".format(ckpt_path))

    cache_key = str(ckpt_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    _ensure_repo_on_path()
    cfg_path = ckpt_path.parent / "cfg.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            "cfg.yaml not found next to checkpoint at {}".format(cfg_path))

    cfg = OmegaConf.load(cfg_path)
    cfg.setdefault("vit_size", "vitl")
    # The demo's argparse defaults that get merged into cfg — we replicate the
    # ones the model actually reads at inference. valid_iters is consumed by
    # the caller, not by __init__.
    cfg.setdefault("hiera", 0)
    cfg.setdefault("low_memory", False)
    cfg.setdefault("scale", 1.0)
    args = OmegaConf.create(cfg)

    from core.foundation_stereo import FoundationStereo  # noqa: E402

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    torch.autograd.set_grad_enabled(False)

    _MODEL_CACHE[cache_key] = (model, args)
    return model, args


def compute_disparity_deep(
    Il_bgr: np.ndarray,
    Ir_bgr: np.ndarray,
    ckpt_dir,
    *,
    iters: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Run FoundationStereo on a TartanAir-style stereo pair.

    Args:
        Il_bgr, Ir_bgr: BGR uint8 arrays of shape ``(H, W, 3)`` (the format
            ``cv2.imread`` returns). FoundationStereo expects RGB float, so we
            swap channels and cast inside.
        ckpt_dir: path to either the model run directory (containing
            ``cfg.yaml`` + ``.pth``) or directly to the ``.pth`` file.
        iters: refinement iteration count. 32 is the demo default; 16 is a
            ~2× speedup with marginal accuracy loss per the paper.
        device: ``"cuda"`` (default) or ``"cpu"`` (extremely slow, debug only).

    Returns:
        ``np.float32`` array of shape ``(H, W)`` — disparity in pixel units.
        Sub-pixel precision preserved (no ``uint8`` clipping).
    """
    import torch

    if Il_bgr.shape != Ir_bgr.shape:
        raise ValueError("left/right images must have identical shape, got {} vs {}".format(
            Il_bgr.shape, Ir_bgr.shape))
    H, W = Il_bgr.shape[:2]

    # _load_model handles the sys.path injection; after this call the
    # FoundationStereo modules are importable from anywhere in this process.
    model, _ = _load_model(ckpt_dir, device=device)
    from core.utils.utils import InputPadder

    # cv2.imread gives BGR uint8; the model expects RGB float (0-255 range,
    # NOT normalized to 0-1 — see normalize_image inside the model).
    Il_rgb = cv2.cvtColor(Il_bgr, cv2.COLOR_BGR2RGB)
    Ir_rgb = cv2.cvtColor(Ir_bgr, cv2.COLOR_BGR2RGB)

    img0 = torch.as_tensor(Il_rgb, device=device).float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(Ir_rgb, device=device).float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0_pad, img1_pad = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(enabled=True):
        disp_pad = model.forward(img0_pad, img1_pad, iters=iters, test_mode=True)
    disp_pad = padder.unpad(disp_pad.float())
    disp = disp_pad.detach().cpu().numpy().reshape(H, W).astype(np.float32)
    return disp


def _self_test() -> None:
    """Quick self-test: load model + run on TartanAir frame 0 + report stats.

    Run via:  python -m bonus.deep_matcher
    """
    import argparse, time
    p = argparse.ArgumentParser(description="FoundationStereo smoke test")
    p.add_argument("--ckpt-dir",
                   default="data/pretrained_models/foundation_stereo/11-33-40",
                   help="directory containing cfg.yaml + model_best_bp2.pth (ViT-small default; "
                        "swap to 23-51-11 for ViT-large)")
    p.add_argument("--left",  default="data/tartanair_raw/japanesealley/Hard/P000/image_left/000000_left.png")
    p.add_argument("--right", default="data/tartanair_raw/japanesealley/Hard/P000/image_right/000000_right.png")
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--save-vis", default="out/deep_matcher_smoke.png",
                   help="where to save a colorized disp PNG")
    args = p.parse_args()

    Il = cv2.imread(args.left)
    Ir = cv2.imread(args.right)
    if Il is None or Ir is None:
        raise SystemExit("[error] failed to read images: {} or {}".format(args.left, args.right))

    print("[run] FoundationStereo on {}x{}  iters={}".format(Il.shape[1], Il.shape[0], args.iters))
    t0 = time.time()
    disp = compute_disparity_deep(Il, Ir, args.ckpt_dir, iters=args.iters)
    t1 = time.time()
    print("[ok]  disp dtype={} shape={} range=[{:.3f}, {:.3f}]  mean={:.3f}".format(
        disp.dtype, disp.shape, float(disp.min()), float(disp.max()), float(disp.mean())))
    print("[ok]  inference {:.2f} s (incl. first-call model build)".format(t1 - t0))

    # Run once more to time steady-state (cached model, no build cost).
    t2 = time.time()
    _ = compute_disparity_deep(Il, Ir, args.ckpt_dir, iters=args.iters)
    t3 = time.time()
    print("[ok]  warm  {:.2f} s/frame".format(t3 - t2))

    if args.save_vis:
        os.makedirs(os.path.dirname(args.save_vis), exist_ok=True)
        vis = cv2.applyColorMap(
            np.clip(disp / max(1.0, float(disp.max())) * 255, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        cv2.imwrite(args.save_vis, vis)
        print("[ok]  vis -> {}".format(args.save_vis))


if __name__ == "__main__":
    _self_test()

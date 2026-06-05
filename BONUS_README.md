# Bonus Verification Guide — NTU CV Spring 2026 Team 10

This bundle is the bonus deliverable for stereo matching + 3D reconstruction.
Everything you need to reproduce our results is included; you do not need to
clone any external repository or have an account on any data hosting service.

---

## 0. TL;DR (three commands after env setup)

```bash
# Set up environment (~5 min)
uv sync --extra bonus --extra deep

# Download the demo dataset (~3 GB) and pretrained weights (~750 MB)
uv run python -m bonus.download_tartanair
uv run python -m bonus.download_fs_weights

# Reproduce the headline result: BPR on the first 100 frames
uv run python -m experiments.eval_tartanair --matcher foundation_stereo --end 100
```

Expected output (last block of the third command):

```
============================================================
Scene:      japanesealley/Hard/P000  frames 0..99
Matcher:    foundation_stereo (deep: data/pretrained_models/foundation_stereo/11-33-40)
Settings:   max_disp=64  threshold=1.0px
------------------------------------------------------------
BPR  mean=3.65%  median=3.85%  worst=13.45%  best=0.74%
Time mean=0.33s  total=33.1s
Valid-mask coverage mean=98.8%
============================================================
```

For the classical baseline number and the 3D walkthrough video, see
[§5](#5-reproduce-bpr-numbers) and [§6](#6-reproduce-the-3d-walkthrough-mp4).

The bundle ships two matchers — that is all you need to verify:

  * **classical** — `computeDisp.py` (the team's Census + WGIF + sub-pixel +
    WMF implementation; selected via `--matcher v8`)
  * **SOTA** — NVIDIA FoundationStereo wrapped in `bonus/deep_matcher.py`
    (selected via `--matcher foundation_stereo`)

---

## 1. Hardware

| Component | Minimum | Recommended |
|---|---|---|
| OS | Linux x86\_64 | Ubuntu 22.04+ |
| Python | 3.10 | 3.11 |
| GPU | none (classical only) | NVIDIA, ≥8 GB VRAM, CUDA 12 |
| Disk | 5 GB (classical only) | 15 GB (deep + dataset + weights) |

Tested on RTX 4090 (24 GB) + Ubuntu 24.04 + CUDA 12.6 + Python 3.11. The
deep matcher (`foundation_stereo`) needs a CUDA GPU; the classical matchers
(`v8`) runs on CPU only.

---

## 2. Network access (no account needed)

The bundle pulls three things from the public internet at runtime. **None of
them require an account, login, or API token.**

| Resource | Endpoint | Anonymous? |
|---|---|---|
| TartanAir dataset (~3 GB) | `https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/...` | yes (public CDN, signed S3 URLs returned to anonymous callers — we verified) |
| FoundationStereo source | already vendored inside this bundle at `third_party/FoundationStereo/` | n/a — offline |
| FoundationStereo weights (~750 MB ViT-small) | `https://drive.google.com/drive/folders/1VhPebc...` | yes (public Google Drive folder, `gdown` works without OAuth) |
| FoundationStereo weights (~3.3 GB ViT-large, optional) | `https://huggingface.co/Felix-Zhenghao/FoundationStereo/...` | yes (public HF mirror, anonymous-readable) |

If your network blocks Google Drive, set `--variant large --fallback-hf` on
the weights downloader to pull only the HuggingFace mirror (see §3.2).

---

## 3. Environment setup

### 3.1 Recommended path: `uv`

[uv](https://docs.astral.sh/uv/) is a fast, deterministic Python package
manager. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the bundle root:

```bash
uv sync --extra bonus --extra deep
```

This creates `.venv/` with everything the bonus pipeline needs (torch 2.4.1
+ CUDA 12, open3d, FoundationStereo runtime deps). On a fresh machine the
torch wheels take ~5 min to download.

### 3.2 Fallback path: pip + venv

If you'd rather not install `uv`, the same dependencies are listed in
`pyproject.toml`. The classical-only setup is small:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirement.txt              # classical track only (~50 MB)
pip install -e ".[bonus,deep]"              # full bonus + deep stack (~3 GB)
```

If pip's PyPI torch is too slow, force the CUDA 12 index:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
pip install -e ".[bonus,deep]"
```

### 3.3 Activate the venv for the rest of this guide

```bash
source .venv/bin/activate
```

All subsequent commands assume the venv is active. With `uv` you can also
prefix any python command with `uv run` instead.

---

## 4. Download the dataset and weights

### 4.1 TartanAir dataset (~3 GB → ~11 GB unzipped)

```bash
python -m bonus.download_tartanair
```

This pulls the default scene `japanesealley/Hard` from the official
HuggingFace mirror — anonymous, no token, supports resumable HTTP `Range`
requests so a network blip just resumes on the next run. Final layout:

```
data/tartanair_raw/japanesealley/Hard/P000/
├── image_left/  000XXX_left.png
├── image_right/ 000XXX_right.png
├── depth_left/  000XXX_left_depth.npy
├── pose_left.txt
└── pose_right.txt
```

To save disk, you can skip depth/right modalities:

```bash
python -m bonus.download_tartanair --modalities image_left,image_right,depth_left
```

The default already covers everything the bonus pipeline + BPR eval need.

### 4.2 FoundationStereo weights (~750 MB ViT-small)

```bash
python -m bonus.download_fs_weights
```

Lands at `data/pretrained_models/foundation_stereo/11-33-40/`
(`cfg.yaml` + `model_best_bp2.pth`).

To verify a previous download instead of fetching:

```bash
python -m bonus.download_fs_weights --check
```

If you also want the ViT-large variant (~3.3 GB, slightly higher accuracy):

```bash
python -m bonus.download_fs_weights --variant large --fallback-hf
```

The `--fallback-hf` flag tells the downloader to grab the big `.pth` from
the HuggingFace community mirror (anonymous) instead of Google Drive, which
is rate-limited for ViT-large.

---

## 5. Reproduce BPR numbers

The eval script computes Middlebury-style bad-pixel ratio against
TartanAir's GT depth, after converting depth to GT disparity via
`d = fx · B / Z`. Valid mask excludes sky (depth ≥ 65000) and pixels whose
GT disparity exceeds `max_disp` (search-range bound).

### 5.1 Both matchers, both thresholds

```bash
for m in v8 foundation_stereo; do
  echo "=== $m @1px ==="
  python -m experiments.eval_tartanair --matcher $m --end 100 --threshold 1.0 | tail -6
  echo "=== $m @3px ==="
  python -m experiments.eval_tartanair --matcher $m --end 100 --threshold 3.0 | tail -6
done
```

### 5.2 Expected results

| Matcher | BPR @ 1 px | BPR @ 3 px | Time / frame |
|---|---|---|---|
| `v8` — classical (`computeDisp.py`: Census + WGIF + sub-pixel + WMF) | **20.26%** | **10.75%** | 0.34 s (CPU) |
| `foundation_stereo` — NVIDIA FoundationStereo ViT-small (zero-shot) | **3.65%** | **1.23%** | 0.28 s (RTX 4090) |

Numbers were generated on RTX 4090 + Ubuntu 24.04 + CUDA 12.6. Classical
time will vary with CPU; deep time will vary with GPU. **BPR numbers
should match to within ±0.1 percentage points** because the deep model is
fully deterministic in eval mode and the classical matcher is
single-precision deterministic.

The 16-percentage-point gap (`20.26%` → `3.65%` at 1\,px) is the headline
result: classical hand-crafted features hit a structural ceiling on
TartanAir's synthetic outdoor textures, and the deep zero-shot prior
breaks through it.

---

## 6. Reproduce the 3D walkthrough mp4

The full pipeline is: stereo pair → disparity → metric depth → masked TSDF
integration → mesh → first-person render along the original capture
trajectory.

```bash
# Classical baseline
python -m bonus.run_bonus_tartanair --matcher v8                --end 100 --out out/v8_demo.mp4

# Deep SOTA (recommended)
python -m bonus.run_bonus_tartanair --matcher foundation_stereo --end 100 --out out/fs_demo.mp4

# Upper-bound reference (uses GT depth instead of any matcher)
python -m bonus.run_bonus_tartanair --use-gt-depth              --end 100 --out out/gt_demo.mp4
```

Each run takes 1–2 min on a 4090 (most of it is the headless Open3D mesh
render). The output mp4s sit in `out/`. Expected file sizes (an indirect
mesh-quality proxy — cleaner mesh compresses smaller):

| Run | mp4 size |
|---|---|
| `v8_demo.mp4` | ~5 MB |
| `fs_demo.mp4` | **~2.7 MB** (close to the GT upper bound) |
| `gt_demo.mp4` | ~2.3 MB |

Open any of them with VLC or `ffplay` to inspect.

---

## 7. Pipeline architecture at a glance

```
                                ┌─────────────────────────┐
   Il, Ir (per frame) ─────────▶│ classical (computeDisp) │──┐
   uint8 BGR 640×480            │ OR deep (deep_matcher)  │  │
                                └─────────────────────────┘  │
                                                             ▼ disparity (float32)
                                                       Z = fx·B / d
                                                             │ depth (metres)
                                                             ▼
                                               sky mask + depth_trunc mask
                                                             │
   TartanAir pose ─── NED→OpenCV ────────▶ Open3D ScalableTSDFVolume.integrate
                                                             │
                                                             ▼
                                                  triangle mesh (marching cubes)
                                                             │
   GT poses ────────────────────▶ render at each original capture pose
                                                             │
                                                             ▼
                                                          out/*.mp4
```

| File | Role |
|---|---|
| `computeDisp.py` | Classical Census + WGIF + sub-pixel + WMF matcher (this IS our team's v8 implementation, shipped here as the canonical classical baseline) |
| `bonus/deep_matcher.py` | FoundationStereo wrapper; `compute_disparity_deep(Il, Ir, ckpt)` |
| `bonus/tartanair.py` | TartanAir V1 loader + **NED→OpenCV pose conversion** |
| `bonus/depth.py` | `Z = fx·B / d` |
| `bonus/fusion.py` | Open3D ScalableTSDFVolume integrate + mesh extract |
| `bonus/render.py` | Follow-trajectory mp4 renderer |
| `bonus/run_bonus_tartanair.py` | The orchestrator; `--matcher {v8,foundation_stereo}` |
| `experiments/eval_tartanair.py` | BPR eval against GT depth |
| `third_party/FoundationStereo/` | NVIDIA's released source (vendored under their LICENSE) |

The classical and deep matchers share a single API contract — only the
`--matcher` flag changes; the downstream depth/TSDF/render code is fully
matcher-agnostic. This means you can verify each matcher in isolation by
swapping the flag.

---

## 8. Troubleshooting

| Symptom | Likely cause + fix |
|---|---|
| `ModuleNotFoundError: torch` | `--extra deep` not synced. Re-run `uv sync --extra bonus --extra deep`. |
| `RuntimeError: CUDA out of memory` | Another process is using the GPU; or try `--matcher-iters 16` on the bonus run. |
| `gdown` fails on FoundationStereo ViT-large | Google Drive's anonymous quota for big files. Use `--variant large --fallback-hf`. |
| HuggingFace TartanAir download stalls | Re-run the same command; `curl -C -` resumes from byte offset. |
| `cv2.imread` returns None on frame N | Dataset extraction was incomplete. Remove `data/tartanair_raw/` and rerun the downloader. |
| `xFormers is not available (...)` warnings during `foundation_stereo` | Harmless — the bundle deliberately skips xFormers; DINOv2 falls back to native attention with no functional change. |
| Open3D renderer says "EGL not available" | The runner needs an OpenGL-capable EGL driver. On headless servers, `nvidia-headless` driver is sufficient; remote SSH typically works. |
| Pose-aligned walkthrough mp4 shows mesh floating above camera path | The NED→OpenCV axis fix in `bonus/tartanair.py:load_poses` got removed. Compare against the line containing `M_NED2CV`. |

---

## 9. Provenance and license

* `PROVENANCE.txt` at the bundle root records the git commit this bundle was
  built from, plus the FoundationStereo source commit. Quote it when
  reporting issues.
* Our own source: see top-level `LICENSE` if present (otherwise: course
  project, all rights reserved).
* FoundationStereo source: NVIDIA Source Code License (Non-Commercial). See
  `third_party/FoundationStereo/LICENSE`. Re-distributed here under the
  redistribution clause (License + copyright notices preserved).
* TartanAir dataset: CC-BY 4.0; downloaded from the official HuggingFace
  mirror at runtime, not bundled.
* FoundationStereo pretrained weights: NVIDIA Source Code License; fetched
  at runtime, not bundled.

---

## 10. Contact

Team 10 — NTU CV Spring 2026.
For any reproduction issue, please attach `PROVENANCE.txt` and the failing
command's full output.

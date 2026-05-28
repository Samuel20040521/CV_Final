"""Bonus track: end-to-end real-world / photoreal stereo pipeline.

Currently active dataset: TartanAir V1 (synthetic, photoreal, with GT depth +
GT pose). Legacy KITTI raw loader is kept under `bonus/kitti.py` / `bonus/
download_sample.py` for the report comparison.

Pipeline shape (dataset-agnostic): loader → calibrated rectification → reuse
`stereo_matching.computeDisp` → disparity-to-depth → TSDF fusion using the
dataset's poses → Open3D flythrough mp4.
"""

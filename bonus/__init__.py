"""Bonus track: end-to-end real-world stereo pipeline on KITTI raw sequences.

Pipeline: KITTI loader → calibrated rectification (from scratch) → reuse
`stereo_matching.computeDisp` → disparity-to-depth → TSDF fusion using KITTI's
GPS/IMU poses → Open3D flythrough mp4.
"""

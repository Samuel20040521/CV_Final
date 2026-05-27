"""v7_mst — Non-local cost aggregation on a Minimum Spanning Tree (Yang 2012).

Build an MST on the 4-connected image graph weighted by color difference, then
recursively aggregate the cost volume through two tree passes (leaf→root,
root→leaf) per disparity slice. This gives non-local support (each pixel sees
the whole image) at near-linear cost in image size.

scipy's MST and BFS routines are written in C, so the heavy graph work is
fast; the per-disparity-slice traversal is pure-numpy with per-node Python
iterations, parallelised across slices via threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .common import _MAX_WORKERS, compute_disp_skeleton


SIGMA = np.float32(0.1)  # Tree-edge weight bandwidth (Yang 2012 default).


def _build_tree(guide_bgr: np.ndarray):
    """Build MST on the 4-connected image graph; return (parents, order, weights).

    parents[i]: index of node i's parent in the tree (or -1 for root)
    order:      BFS order (root first) — 1-D array of node indices
    weights[i]: edge weight = exp(-||I[i] - I[parents[i]]|| / SIGMA), or 0 for root
    """
    H, W, _ = guide_bgr.shape
    N = H * W
    I = guide_bgr.astype(np.float32) / 255.0  # (H, W, 3)

    # 4-neighbor edges as (row, col, color_dist) arrays.
    # Right edges: (y, x) → (y, x+1), color diff = ||I[y,x] - I[y,x+1]||
    diff_right = np.linalg.norm(I[:, 1:, :] - I[:, :-1, :], axis=2)  # (H, W-1)
    # Down edges: (y, x) → (y+1, x)
    diff_down = np.linalg.norm(I[1:, :, :] - I[:-1, :, :], axis=2)   # (H-1, W)

    # Linear indices.
    yy, xx = np.meshgrid(np.arange(H), np.arange(W - 1), indexing="ij")
    src_r = (yy * W + xx).ravel()
    dst_r = (yy * W + xx + 1).ravel()
    w_r = diff_right.ravel()

    yy, xx = np.meshgrid(np.arange(H - 1), np.arange(W), indexing="ij")
    src_d = (yy * W + xx).ravel()
    dst_d = ((yy + 1) * W + xx).ravel()
    w_d = diff_down.ravel()

    rows = np.concatenate([src_r, src_d])
    cols = np.concatenate([dst_r, dst_d])
    data = np.concatenate([w_r, w_d]).astype(np.float32) + 1e-6  # avoid zero edges

    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    mst = minimum_spanning_tree(graph)
    # Symmetrise (BFS needs undirected).
    mst_sym = mst + mst.T

    order, predecessors = breadth_first_order(mst_sym, i_start=0, return_predecessors=True)
    parents = predecessors.astype(np.int64)  # -9999 for unreached, but tree should reach all

    # Build edge color distance for each node→parent edge.
    # We have color diffs stored implicitly via I; recompute for (i, parents[i]).
    I_flat = I.reshape(N, 3)
    has_parent = parents >= 0
    diffs = np.zeros(N, dtype=np.float32)
    diffs[has_parent] = np.linalg.norm(
        I_flat[has_parent] - I_flat[parents[has_parent]], axis=1
    )
    weights = np.exp(-diffs / SIGMA).astype(np.float32)
    # Root edge weight is undefined — set 0 so it contributes nothing.
    weights[~has_parent] = 0.0

    return parents, order, weights


def _mst_aggregate_slice(cost_flat_d: np.ndarray, parents: np.ndarray,
                         order: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Single-slice MST aggregation. cost_flat_d shape (N,).

    Two-pass recursion (Yang 2012):
      Up:    C_up[parent] += s(child, parent) * C_up[child]
      Down:  C_out[v]     = s(v, parent) * (C_out[parent] - s(v, parent) * C_up[v]) + C_up[v]
    """
    C_up = cost_flat_d.copy()
    # Upward pass: process nodes leaf-first (reverse BFS order).
    for i in order[::-1]:
        p = parents[i]
        if p >= 0:
            C_up[p] += weights[i] * C_up[i]

    C_out = C_up.copy()
    # Downward pass: skip root (first in order).
    for i in order[1:]:
        p = parents[i]
        w = weights[i]
        C_out[i] = w * (C_out[p] - w * C_up[i]) + C_up[i]
    return C_out


def aggregate(cost_volume: np.ndarray, guide_bgr: np.ndarray) -> np.ndarray:
    D, H, W = cost_volume.shape
    N = H * W

    parents, order, weights = _build_tree(guide_bgr)
    cost_flat = cost_volume.reshape(D, N)
    out_flat = np.empty_like(cost_flat)

    def filter_slice(d: int) -> None:
        out_flat[d] = _mst_aggregate_slice(cost_flat[d], parents, order, weights)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        list(ex.map(filter_slice, range(D)))
    return out_flat.reshape(D, H, W)


def computeDisp(Il: np.ndarray, Ir: np.ndarray, max_disp: int) -> np.ndarray:
    return compute_disp_skeleton(Il, Ir, max_disp, aggregate)

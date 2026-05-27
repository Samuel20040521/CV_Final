import os
import numpy as np
import cv2
import cv2.ximgproc as xip
import concurrent.futures

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape

    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # Step 1: Cost Computation
    # #gray
    gray_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(np.float32)

    window_size = 5
    half_w = window_size // 2

    bits_L = np.zeros((h, w, window_size**2 - 1), dtype=np.bool_)
    bits_R = np.zeros((h, w, window_size**2 - 1), dtype=np.bool_)

    padded_L = np.pad(gray_L, half_w, mode='edge')
    padded_R = np.pad(gray_R, half_w, mode='edge')

    idx = 0
    for dy in range(-half_w, half_w + 1):
        for dx in range(-half_w, half_w + 1):
            if dy == 0 and dx == 0:
                continue

            shifted_L = padded_L[half_w + dy : h + half_w + dy, half_w + dx : w + half_w + dx]
            shifted_R = padded_R[half_w + dy : h + half_w + dy, half_w + dx : w + half_w + dx]

            bits_L[:, :, idx] = (shifted_L >= gray_L)
            bits_R[:, :, idx] = (shifted_R >= gray_R)
            idx += 1


    cost_L = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_R = np.zeros((h, w, max_disp), dtype=np.float32)

    def compute_cost_d(d):
        if d == 0:
            c_L_d = np.sum(bits_L != bits_R, axis=2)
            c_R_d = c_L_d.copy()
        else:

            diff_L = bits_L[:, d:, :] != bits_R[:, :-d, :]
            c_L_d = np.zeros((h, w), dtype=np.float32)
            c_L_d[:, d:] = np.sum(diff_L, axis=2)
            c_L_d[:, :d] = c_L_d[:, d].reshape(-1, 1)

            diff_R = bits_R[:, :-d, :] != bits_L[:, d:, :]
            c_R_d = np.zeros((h, w), dtype=np.float32)
            c_R_d[:, :-d] = np.sum(diff_R, axis=2)
            c_R_d[:, w-d:] = c_R_d[:, w-d-1].reshape(-1, 1)
        return d, c_L_d, c_R_d

    available_cores = os.cpu_count() or 4

    # Cost Volume
    with concurrent.futures.ThreadPoolExecutor(max_workers=available_cores) as executor:
        futures = [executor.submit(compute_cost_d, d) for d in range(max_disp)]
        for future in concurrent.futures.as_completed(futures):
            d, c_L_d, c_R_d = future.result()
            cost_L[:, :, d] = c_L_d
            cost_R[:, :, d] = c_R_d

    # Step 2: Cost Aggregation (Guided Filter)
    radius = 7
    eps = 10.0 

    def aggregate_d(d):
        c_L = cost_L[:, :, d]
        c_R = cost_R[:, :, d]
        agg_L = xip.guidedFilter(guide=Il, src=c_L, radius=radius, eps=eps)
        agg_R = xip.guidedFilter(guide=Ir, src=c_R, radius=radius, eps=eps)
        return d, agg_L, agg_R

    # Guided Filter
    with concurrent.futures.ThreadPoolExecutor(max_workers=available_cores) as executor:
        futures = [executor.submit(aggregate_d, d) for d in range(max_disp)]
        for future in concurrent.futures.as_completed(futures):
            d, agg_L, agg_R = future.result()
            cost_L[:, :, d] = agg_L
            cost_R[:, :, d] = agg_R


    # Step 3: Disparity Optimization
    disp_L = np.argmin(cost_L, axis=2).astype(np.float32)
    disp_R = np.argmin(cost_R, axis=2).astype(np.float32)


    # Step 4: Disparity Refinement
    y, x = np.mgrid[0:h, 0:w]
    x_R = x - disp_L
    x_R = np.clip(x_R, 0, w - 1).astype(int) 
    disp_R_mapped = disp_R[y, x_R]
    
    valid_mask = disp_L == disp_R_mapped

    D = disp_L.copy()
    D[~valid_mask] = np.inf

    F_L = np.copy(D)
    F_R = np.copy(D)
    
    F_L[:, 0] = np.where(F_L[:, 0] == np.inf, max_disp, F_L[:, 0])

    for j in range(1, w):
        F_L[:, j] = np.where(F_L[:, j] == np.inf, F_L[:, j-1], F_L[:, j])

    F_R[:, -1] = np.where(F_R[:, -1] == np.inf, max_disp, F_R[:, -1])

    for j in range(w - 2, -1, -1):
        F_R[:, j] = np.where(F_R[:, j] == np.inf, F_R[:, j+1], F_R[:, j])
                
    D_filled = np.minimum(F_L, F_R)
    labels = cv2.medianBlur(D_filled.astype(np.uint8), 3)
    labels = xip.weightedMedianFilter(joint=Il.astype(np.uint8), src=labels, r=9)

    return labels.astype(np.uint8)
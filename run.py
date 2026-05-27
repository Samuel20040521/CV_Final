import argparse
import glob
import os
import time
import cv2
import numpy as np

# 從你的實作檔案中匯入 computeDisp
from computeDisp import computeDisp
from eval import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Run all Middlebury datasets and compute ranking score"
    )
    parser.add_argument(
        "--dataset_path", default="./testdata/", help="path to testing dataset"
    )
    args = parser.parse_args()

    # 4組影像的設定 (max_disp, scale_factor, threshold_limit)
    config = {
        "Tsukuba": (15, 16, 8.0),
        "Venus": (20, 8, 5.0),
        "Teddy": (60, 4, 18.0),
        "Cones": (60, 4, 15.0)
    }

    images = ["Tsukuba", "Venus", "Teddy", "Cones"]

    # 用來紀錄每張圖的統計數據
    results = {}
    total_time = 0.0
    bpr_product = 1.0

    print("=" * 60)
    print("  Stereo Matching Pipeline - Batch Evaluation")
    print("=" * 60)

    for img_name in images:
        print(f"👉 Processing {img_name}...")

        # 1. 讀取影像
        img_left = cv2.imread(
            os.path.join(args.dataset_path, img_name, "img_left.png")
        )
        img_right = cv2.imread(
            os.path.join(args.dataset_path, img_name, "img_right.png")
        )

        max_disp, scale_factor, bpr_limit = config[img_name]

        # 2. 執行立體匹配並計時
        t0 = time.time()
        labels = computeDisp(img_left, img_right, max_disp)
        elapsed_time = time.time() - t0

        total_time += elapsed_time

        # 3. 讀取 Ground Truth 並評估
        gt_path = glob.glob(
            os.path.join(args.dataset_path, img_name, "disp_gt.*")
        )
        if len(gt_path) != 0:
            img_gt = cv2.imread(gt_path[0], -1)
            # 取得原始機率值 (0.0 ~ 1.0)
            error_ratio = evaluate(labels, img_gt, scale_factor)
            bpr_percent = error_ratio * 100.0
        else:
            print(f"⚠️  [Warning] Ground truth for {img_name} not found!")
            bpr_percent = 0.0
            error_ratio = 0.0

        # 計算 BPR 的乘積 (評分標準公式)
        bpr_product *= bpr_percent

        # 紀錄結果
        results[img_name] = {
            "time": elapsed_time,
            "bpr": bpr_percent,
            "limit": bpr_limit,
        }

        # 即時印出單張結果
        status = "✅ PASS" if bpr_percent < bpr_limit else "❌ FAIL"
        print(f"   [{status}] Time: {elapsed_time:.4f} sec | Bad Pixel Ratio: {bpr_percent:.2f}% (Target: <{bpr_limit}%)")
        if elapsed_time > 600:
            print("   ⚠️  [TIMEOUT] Exceeded 10 minutes limit!")
        print("-" * 60)

    # 4. 計算最終總體 Ranking 分數
    ranking_score = bpr_product * total_time

    # =============================================================================
    # 顯示最終統計總表
    # =============================================================================
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Image':<10} | {'Time (sec)':<12} | {'BPR (%)':<10} | {'Target (%)':<10}")
    print("-" * 60)
    for img_name in images:
        print(
            f"{img_name:<10} | {results[img_name]['time']:<12.4f} | {results[img_name]['bpr']:<10.2f} | <{results[img_name]['limit']:<10.1f}"
        )
    print("-" * 60)
    print(f"⏱️  Total Execution Time : {total_time:.4f} sec")
    print(f"📈 BPR Product (Π BPR_i): {bpr_product:.4f}")
    print(f"🏆 Final Ranking Score : {ranking_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
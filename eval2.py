import numpy as np
import cv2
import argparse
import time
import os, glob
from computeDisp_v6 import computeDisp


def evaluate(disp_input, disp_gt, scale_factor, threshold=1.0):
    disp_input = np.uint8(disp_input * scale_factor)
    disp_input = np.int32(disp_input/scale_factor)
    disp_gt = np.int32(disp_gt/scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1

    return float(nr_error)/nr_pixel


def evaluate_image(image_name, dataset_path, config):
    print('Processing image %s ...' % image_name)
    t0 = time.time()
    img_left = cv2.imread(os.path.join(dataset_path, image_name, 'img_left.png'))
    img_right = cv2.imread(os.path.join(dataset_path, image_name, 'img_right.png'))
    if img_left is None or img_right is None:
        raise FileNotFoundError(f'Could not load image pair for {image_name} in {dataset_path}')

    max_disp, scale_factor = config[image_name]
    labels = computeDisp(img_left, img_right, max_disp)
    elapsed = time.time() - t0
    print('[Time] %.4f sec' % elapsed)

    gt_path = glob.glob(os.path.join(dataset_path, image_name, 'disp_gt.*'))
    bpr = None
    if len(gt_path) != 0:  # gt exists
        img_gt = cv2.imread(gt_path[0], -1)
        bpr = evaluate(labels, img_gt, scale_factor)
        print('[Bad Pixel Ratio] %.2f%%' % (bpr*100))
    else:
        print(f'[Warning] No ground truth found for {image_name}')

    return {
        'image': image_name,
        'time': elapsed,
        'bpr': bpr,
    }


def score_results(results, image_names):
    bpr_product = 1.0
    total_time = 0.0
    for image_name in image_names:
        entry = results.get(image_name)
        if entry is None or entry['bpr'] is None:
            raise ValueError(f'Missing BPR result for {image_name}')
        # Multiply percentage values directly (e.g. 10 * 20), not fractional ratios.
        bpr_product *= entry['bpr'] * 100.0
        total_time += entry['time']
    return bpr_product * total_time


def main():
    parser = argparse.ArgumentParser(description='evaluation function of stereo matching (batch)')
    parser.add_argument('--dataset_path', default='./testdata/', help='path to testing dataset')
    args = parser.parse_args()

    config = {
        'Tsukuba': (15, 16),
        'Venus':   (20, 8),
        'Teddy':   (60, 4),
        'Cones':   (60, 4),
    }

    results = {}
    for image_name in ['Tsukuba', 'Venus', 'Teddy', 'Cones']:
        results[image_name] = evaluate_image(image_name, args.dataset_path, config)

    print('\nSummary:')
    for image_name in ['Tsukuba', 'Venus', 'Teddy', 'Cones']:
        entry = results[image_name]
        print('  %s: time=%.4f sec, BPR=%.2f%%' % (image_name, entry['time'], entry['bpr'] * 100))

    score_all = score_results(results, ['Tsukuba', 'Venus', 'Teddy', 'Cones'])
    score_no_cones = score_results(results, ['Tsukuba', 'Venus', 'Teddy'])

    print('\nScore (all 4 images): %.8f' % score_all)
    print('Score (Tsukuba, Venus, Teddy): %.8f' % score_no_cones)


if __name__ == '__main__':
    main()

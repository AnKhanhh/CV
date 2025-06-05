import itertools
import os
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
import misc

# Harris original paper implementation: Gaussian neighborhood window, 3x3 Sobel aperture
# Opencv's Harris: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/corner.cpp#L634

# Dictionary of image gradient kernels
gradient_kernels = {
    "prewitt3x3": {
        "kernel": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]),
        "normalize": 1 / 6
    },
    "sobel3x3": {
        "kernel": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        "normalize": 1 / 8
    },
    "scharr3x3": {
        "kernel": np.array([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ]),
        "normalize": 1 / 32
    },
    "prewitt5x5": {
        "kernel": np.array([
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]
        ]),
        "normalize": 1 / 50
    },
    "sobel5x5": {
        "kernel": np.array([
            [-5, -4, 0, 4, 5],
            [-8, -10, 0, 10, 8],
            [-10, -20, 0, 20, 10],
            [-8, -10, 0, 10, 8],
            [-5, -4, 0, 4, 5]
        ]),
        "normalize": 1 / 240
    },
    "scharr5x5": {
        "kernel": np.array([
            [-1, -1, 0, 1, 1],
            [-2, -2, 0, 2, 2],
            [-3, -6, 0, 6, 3],
            [-2, -2, 0, 2, 2],
            [-1, -1, 0, 1, 1]
        ]),
        "normalize": 1 / 60
    }
}


def harris_core(input_img: np.ndarray,
                aperture_size: int, aperture_type: str, border_type: int,
                k: float, window_size: int, window_type: str) -> np.ndarray:
    # 1. Calculate gradients and products
    kernel_info = gradient_kernels[f"{aperture_type}{aperture_size}x{aperture_size}"]
    kx = kernel_info["kernel"] * kernel_info["normalize"]
    ky = kx.T
    dx = cv2.filter2D(input_img, -1, kx)
    dy = cv2.filter2D(input_img, -1, ky)
    i_xx = dx * dx
    i_xy = dx * dy
    i_yy = dy * dy

    # 2. Sum gradient products
    if window_type == 'gaussian':
        kernel_1d = cv2.getGaussianKernel(window_size, 0)  # sigma = 0.3*((ksize-1)*0.5 – 1) + 0.8
    elif window_type == 'binary' or window_type == 'uniform':
        kernel_1d = np.ones(window_size) / window_size
    else:
        raise Exception("unknown window type")
    sum_i_xx = cv2.sepFilter2D(i_xx, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)
    sum_i_xy = cv2.sepFilter2D(i_xy, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)
    sum_i_yy = cv2.sepFilter2D(i_yy, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)

    # 3.Calculate Harris response
    det = (sum_i_xx * sum_i_yy) - (sum_i_xy * sum_i_xy)
    trace = sum_i_xx + sum_i_yy
    harris_response = det - k * (trace * trace)

    return harris_response


# noinspection DuplicatedCode
def harris_pipeline(input_img: np.ndarray,
                    aperture_size: int = 3, aperture_type: str = 'sobel', border_type: int = cv2.BORDER_REFLECT_101,
                    k: float = 0.04, window_size: int = 5, window_type: str = 'gaussian',
                    threshold_ratio: float = 0.01, visualize: bool = False) -> Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray]:
    """
    Detect Harris corners in the input grayscale image

    Returns:
        List of detected corners as (x, y, response_value),
        Original image with detected corners marked in red,
        Heatmap of Harris corner response
    """
    # 1. Param validation
    window_size = window_size | 1  # always odd
    aperture_size = max(min(aperture_size | 1, 5), 3)  # only 3 or 5
    window_type = window_type.lower()  # always lowercase
    aperture_type = aperture_type.lower()
    if len(input_img.shape) == 3:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if input_img.dtype != np.float32:
        input_img = input_img.astype(np.float32)

    # 2. Obtain response map
    harris_response = harris_core(input_img, aperture_size, aperture_type, border_type, k, window_size, window_type)

    # 3. NMS and thresholding
    # For manual implementation: dilation NMS -> thresholding -> dist filtering
    nms_radius = int(window_size / 2) + 1
    coordinates: Tuple[List[int], List[int]] = peak_local_max(harris_response, min_distance=nms_radius, threshold_rel=threshold_ratio)

    # 4. Post-processing steps
    # Refine corners to subpixel
    corner_list: List[Tuple[int, int, float]] = [(x, y, harris_response[y, x]) for y, x in coordinates]
    # points = np.float32([corner[:2] for corner in corner_list])
    # refined_coords = cv2.cornerSubPix(input_img, points, (5, 5), (-1, -1),
    #                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # refined_corner_list = [(point[0], point[1], corner_list[i][2])
    #                        for i, point in enumerate(refined_coords)]

    if visualize:
        # Create corner marking image
        corner_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        max_response: float = max(score for _, _, score in corner_list) or 1.0
        for x, y, score in corner_list:
            radius = int(3 + (score / max_response) * 10)
            cv2.circle(corner_img, (x, y), radius, (255, 0, 0), 1)

        # Create response heatmap
        norm_response = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)
    else:
        corner_img = None
        heatmap = None

    return corner_list, corner_img, heatmap


def find_corner_matches(reference_corners, distorted_corners, max_dist_pixel=3.0):
    """
    Find optimal matches between reference and distorted corners
    """
    # Create distance matrix
    cost_matrix = np.zeros((len(reference_corners), len(distorted_corners)))
    for i, (x1, y1, _) in enumerate(reference_corners):
        for j, (x2, y2, _) in enumerate(distorted_corners):
            cost_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # Match using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Filter valid matches
    valid_matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < max_dist_pixel]

    return valid_matches, cost_matrix


def calculate_repeatability(reference_corners, distorted_corners, matches):
    """
    Calculate repeatability score (0-1)
    """
    ref_count = len(reference_corners)
    dist_count = len(distorted_corners)
    return 0.0 if min(ref_count, dist_count) == 0 else len(matches) / ref_count


def calculate_localization_accuracy(matches, cost_matrix) -> float:
    """
    Calculate average distance between matched corners
    """
    return float('inf') if not matches else sum(cost_matrix[i, j] for i, j in matches) / len(matches)


# noinspection PyUnresolvedReferences
def response_degradation(reference_corners, distorted_corners, matches) -> float:
    """
    Calculate the average ratio of response value between distorted and reference image
    """
    if not matches:
        return float('nan')

    ratios = []
    for i, j in matches:
        ref_resp = reference_corners[i][2]
        dist_resp = distorted_corners[j][2]
        ratios.append(dist_resp / ref_resp)
    return np.mean(ratios).item()


def process_single_harris_parameter_set(param_set, distortion_no, ref_img_list):
    """Process a single Harris parameter combination across all distortion levels."""
    win_sz, win_tp, ap_sz, ap_tp, k = param_set

    # Compute reference corners once per parameter set
    ref_corners_dict = _compute_harris_reference_corners(
        ref_img_list, win_sz, win_tp, ap_sz, ap_tp, k
    )

    # Process all distortion levels for this parameter set
    results = []
    for level in range(1, 6):
        result = _process_harris_distortion_level(
            distortion_no, level, ref_corners_dict,
            win_sz, win_tp, ap_sz, ap_tp, k
        )
        results.append(result)

    return results


def _compute_harris_reference_corners(ref_img_list, win_sz, win_tp, ap_sz, ap_tp, k):
    """Compute Harris corners for all reference images with given parameters."""
    ref_corners_dict = {}
    for img_no in ref_img_list:
        ref_image_path = f"tid2013/reference_images/I{img_no}.BMP"
        ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
        if ref_image is None:
            continue

        ref_corners, *_ = harris_pipeline(
            ref_image, k=k, window_size=win_sz, aperture_size=ap_sz,
            aperture_type=ap_tp, window_type=win_tp
        )
        ref_corners_dict[img_no] = ref_corners
    return ref_corners_dict


def _process_harris_distortion_level(distortion_no, level, ref_corners_dict,
                                     win_sz, win_tp, ap_sz, ap_tp, k):
    """Process a single distortion level for given Harris parameters."""
    # Initialize accumulators
    total_ref_corners = total_matches = total_images = 0
    sum_repeatability = 0.0
    loc_distances, resp_ratios = [], []

    for img_no, ref_corners in ref_corners_dict.items():
        # Load and process distorted image
        dist_image_path = f"tid2013/distorted_images/i{img_no}_{distortion_no}_{level}.bmp"
        dist_image = cv2.imread(dist_image_path, cv2.IMREAD_COLOR)
        if dist_image is None:
            continue

        dist_corners, *_ = harris_pipeline(
            dist_image, aperture_size=ap_sz, aperture_type=ap_tp,
            window_size=win_sz, window_type=win_tp, k=k
        )

        matches, cost_matrix = find_corner_matches(ref_corners, dist_corners)

        # Update counters
        total_ref_corners += len(ref_corners)
        total_matches += len(matches)
        total_images += 1

        if len(ref_corners) > 0:
            sum_repeatability += len(matches) / len(ref_corners)

        # Collect match metrics
        for i_ref, i_dist in matches:
            loc_distances.append(float(cost_matrix[i_ref, i_dist]))
            ref_resp, dist_resp = ref_corners[i_ref][2], dist_corners[i_dist][2]
            resp_ratios.append(float(dist_resp / ref_resp))

    # Compute aggregated metrics
    mean_repeatability = sum_repeatability / total_images if total_images > 0 else 0.0
    mean_localization = np.mean(loc_distances) if loc_distances else float('inf')
    mean_resp_ratio = np.mean(resp_ratios) if resp_ratios else float('nan')

    return {
        'dt_no': int(distortion_no), 'dt_lv': int(level),
        'window_sz': float(win_sz),
        'window_tp': {'uniform': 0, 'gaussian': 1}.get(win_tp),
        'aperture_sz': float(ap_sz),
        'aperture_tp': {'sobel': 0, 'prewitt': 1, 'scharr': 2}.get(ap_tp),
        'k_val': float(k),
        'total_ref_corners': int(total_ref_corners), 'total_matches': int(total_matches),
        'sampling_num': int(total_images), 'mean_repeatability': float(mean_repeatability),
        'mean_localization': float(mean_localization),
        'mean_resp_ratio': float(mean_resp_ratio),
    }


def harris_wrapper(distortion_no: str, ref_img_list: List[str],
                   win_size_list: Tuple = (3, 5, 7, 9),
                   win_type_list: Tuple = ("gaussian", "uniform"),
                   aperture_size_list: Tuple = (3, 5),
                   aperture_type_list: Tuple = ("sobel", "prewitt", "scharr"),
                   k_list: Tuple = (0.04, 0.06, 0.12, 0.16),
                   n_jobs=-1):
    # Generate parameter combinations
    param_combinations = list(itertools.product(
        win_size_list, win_type_list, aperture_size_list, aperture_type_list, k_list
    ))

    print(f"Processing {len(param_combinations)} parameter sets for distortion #{distortion_no} "
          f"across {len(ref_img_list)} reference images using {n_jobs} jobs")

    # Parallel processing of parameter sets
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_harris_parameter_set)(param_set, distortion_no, ref_img_list)
        for param_set in param_combinations
    )

    # Flatten results (each parameter set returns 5 distortion levels)
    flattened_results = [result for param_results in all_results for result in param_results]

    # Create and save dataframe
    df = pd.DataFrame(flattened_results)

    os.makedirs(f"results/distortion_{distortion_no}", exist_ok=True)
    output_filename = f"results/distortion_{distortion_no}/harris_metrics_dist{distortion_no}.parquet"
    df.to_parquet(output_filename)

    print(f"Data exported to {output_filename}")
    return df


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()

    # brick wall #1: sharp corner, homogeneous, high texture
    # human face #4: soft corner, low contrast
    # house row #8: sharp corner, high + low contrast
    # ship sails #9: soft corner, high texture, varying scales
    # grass + rocks #13: Heterogeneous, high texture
    # fence + grass #19: edge-dominant, texture + smooth
    ref_img_list = ["01", "04", "08", "09", "13", "19"]
    _ = harris_wrapper("01", ref_img_list, n_jobs=14)

    end_time = time.perf_counter()
    print(f"=== Executed in {(end_time - start_time):.1f}s ===")

import itertools
import os
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment

# Harris original paper implementation: Gaussian neighborhood window, 3x3 Sobel aperture
# Opencv's Harris: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/corner.cpp#L634

# Dictionary of image gradient kernels
gradient_kernels = {
    "prewitt3x3": {
        "kernel": np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]),
        "normalize": 1 / 6
    },
    "sobel3x3": {
        "kernel": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        "normalize": 1 / 8
    },
    "scharr3x3": {
        "kernel": np.array([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
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
                    threshold_ratio: float = 0.01, nms_radius: int = 10,
                    visualize: bool = False) -> Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray]:
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
    coordinates: Tuple[List[int], List[int]] = peak_local_max(harris_response, min_distance=nms_radius, threshold_rel=threshold_ratio)

    # 4. Post-processing steps
    # Refine corners to subpixel
    corner_list: List[Tuple[int, int, float]] = [(x, y, harris_response[y, x]) for y, x in coordinates]
    points = np.float32([corner[:2] for corner in corner_list])
    subp_win_size = window_size * 2 - 1
    refined_points = cv2.cornerSubPix(input_img, points, (subp_win_size, subp_win_size), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    refined_corner_list = [(point[0], point[1], corner_list[i][2])
                           for i, point in enumerate(refined_points)]

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

    return refined_corner_list, corner_img, heatmap


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


def evaluate_corner_detection(reference_corners, distorted_corners, max_dist_pixel=3.0):
    """
    Evaluate corner detection performance using all metrics
    Returns:
        Dictionary of performance metrics
    """
    # Find matches
    matches, cost_matrix = find_corner_matches(reference_corners, distorted_corners, max_dist_pixel)

    # Calculate metrics
    repeatability = calculate_repeatability(reference_corners, distorted_corners, matches)
    localization = calculate_localization_accuracy(matches, cost_matrix)
    response_analysis = response_degradation(reference_corners, distorted_corners, matches)

    return {
        "repeatability": repeatability,
        "localization_accuracy": localization,
        "response_degradation": response_analysis,
        "num_matches": len(matches)
    }


def harris_wrapper(test_config,
                   win_size_list: Tuple[int] = (3, 5, 7, 9),
                   win_type_list: Tuple[str] = ("gaussian", "uniform"),
                   aperture_size_list: Tuple[int] = (3, 5),
                   aperture_type_list: Tuple[str] = ("sobel", "prewitt", "scharr"),
                   k_list: Tuple[float] = (0.04, 0.06, 0.12, 0.16)):
    """
    Evaluate Harris performance with parameters across distortion levels.
    Returns:
        DataFrame containing all performance metrics
    """
    img_no, distortion_no = test_config.split(":")

    # Load reference image
    ref_image_path = f"tid2013/reference_images/I{img_no}.BMP"
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    if ref_image is None:
        print(f"Error loading image at {ref_image_path}")
        return None

    total_iter = len(win_size_list) * len(aperture_size_list) * len(k_list) * len(win_type_list) * len(aperture_type_list)
    print(f"Calculating performance metrics of {total_iter} parameter sets, for image #{img_no} distortion #{distortion_no}:")

    # For each parameter combination
    results_df = []
    for i, (win_sz, win_tp, ap_sz, ap_tp, k) in enumerate(itertools.product(win_size_list, win_type_list, aperture_size_list, aperture_type_list, k_list)):
        ref_corners, *_ = harris_pipeline(ref_image, k=k, window_size=win_sz, aperture_size=ap_sz)
        print(f"i{i + 1}..", end="✓  ")

        # For each distortion level
        for level in (1, 2, 3, 4, 5):
            # Load distorted image
            dist_image_path = f"tid2013/distorted_images/i{img_no}_{distortion_no}_{level}.bmp"
            dist_image = cv2.imread(dist_image_path, cv2.IMREAD_COLOR)
            if dist_image is None:
                print(f"Error loading image at {dist_image_path}")
                continue
            dist_corners, *_ = harris_pipeline(dist_image,
                                               aperture_size=ap_sz, aperture_type=ap_tp,
                                               window_size=win_sz, window_type=win_tp,
                                               k=k)

            # Calculate metrics
            metrics = evaluate_corner_detection(ref_corners, dist_corners)
            result_row = {
                'rf_img_no': int(img_no),  # image no
                'dt_no': int(distortion_no),  # distortion no
                'dt_lv': int(level),  # distortion level
                'window_sz': float(win_sz),  # harris window size
                'windows_tp': {'uniform': 0, 'gaussian': 1}.get(win_tp),  # uniform=0, gaussian=1
                'aperture_sz': float(ap_sz),  # gradient window size
                'aperture_tp': {'sobel': 0, 'prewitt': 1, 'scharr': 2}.get(ap_tp),  # sobel=0, prewitt=1, scharr=2
                'k_val': float(k),  # harris k value
                'corner_matches': int(metrics['num_matches']),  # matches between 2 images
                'repeatability': metrics['repeatability'],  # ratio of matches / reference corners
                'localization_acc': metrics['localization_accuracy'],  # avg distance between matched corners, in px
                'mean_resp_ratio': metrics['response_degradation']  # avg ratio of distorted / reference response in matched corners
            }
            results_df.append(result_row)

    if not results_df:
        print("No results collected. Check file paths and parameters.")
        return None

    # Convert results to DataFrame
    df = pd.DataFrame(results_df)
    os.makedirs(f"results/distortion_{distortion_no}", exist_ok=True)
    output_filename = f"results/distortion_{distortion_no}/harris_img{img_no}_dist{distortion_no}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

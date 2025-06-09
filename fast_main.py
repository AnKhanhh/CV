import itertools
import os
from typing import List, Tuple, Dict

import numpy as np
import cv2
import pandas as pd
from joblib import Parallel, delayed
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max


def generate_circle_offsets_cached(radius, _cache={}):
    """Cached version of circle offset generation"""
    if radius not in _cache:
        rr, cc = circle_perimeter(0, 0, radius)
        offsets = np.unique(np.column_stack([rr, cc]), axis=0).tolist()
        offsets.sort(key=lambda p: np.arctan2(p[1], p[0]))
        _cache[radius] = offsets
    return _cache[radius]


def get_cardinal_indices(offsets, radius):
    """Get indices of cardinal points in offset array"""
    cardinals = [(radius, 0), (0, radius), (-radius, 0), (0, -radius)]
    indices = []
    for cardinal in cardinals:
        distances = [abs(p[0] - cardinal[0]) + abs(p[1] - cardinal[1]) for p in offsets]
        indices.append(np.argmin(distances))
    return indices


def extract_circle_intensities_vectorized(image, offsets, radius):
    """Vectorized extraction of circle intensities for all valid pixels"""
    h, w = image.shape

    # Valid pixel region
    y_start, y_end = radius, h - radius
    x_start, x_end = radius, w - radius

    # Create meshgrid for all valid coordinates
    y_coords, x_coords = np.meshgrid(
        np.arange(y_start, y_end),
        np.arange(x_start, x_end),
        indexing='ij'
    )

    # Flatten for vectorized operations
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()
    n_pixels = len(y_flat)

    # Extract center intensities
    center_intensities = image[y_flat, x_flat].astype(np.float32)

    # Extract circle intensities
    n_offsets = len(offsets)
    circle_intensities = np.zeros((n_pixels, n_offsets), dtype=np.float32)

    for i, (dy, dx) in enumerate(offsets):
        circle_intensities[:, i] = image[y_flat + dy, x_flat + dx]

    return center_intensities, circle_intensities, y_coords, x_coords


def vectorized_high_speed_test(center_intensities, circle_intensities,
                               cardinal_indices, threshold, n_ratio):
    """Vectorized high speed test"""
    cardinal_intensities = circle_intensities[:, cardinal_indices]
    differences = cardinal_intensities - center_intensities[:, np.newaxis]

    bright_mask = differences > threshold
    dark_mask = differences < -threshold

    max_count = np.maximum(bright_mask.sum(axis=1), dark_mask.sum(axis=1))
    required_cardinal = max(1, int(n_ratio / 0.25))  # Match your original logic

    return max_count >= required_cardinal


def find_longest_arc_optimized(mask):
    """Optimized arc detection using run-length encoding"""
    if not mask.any():
        return []

    n = len(mask)
    if mask.all():
        return list(range(n))

    # Efficient run detection using diff
    doubled = np.concatenate([mask, mask])
    padded = np.pad(doubled, 1, constant_values=False)
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return []

    lengths = ends - starts
    valid_runs = [(s, l) for s, l in zip(starts, lengths) if l <= n]

    if not valid_runs:
        return []

    start, length = max(valid_runs, key=lambda x: x[1])
    return [(start + i) % n for i in range(length)]


def vectorized_arc_and_corner(center_intensities, circle_intensities,
                              threshold, required_n, cornerness_calculation):
    """Vectorized version of arc_and_corner with preserved logic"""
    n_pixels = len(center_intensities)
    cornerness_scores = np.zeros(n_pixels, dtype=np.float32)

    differences = circle_intensities - center_intensities[:, np.newaxis]
    bright_mask = differences > threshold
    dark_mask = differences < -threshold

    cornerness_calculation = cornerness_calculation.lower()

    # Process each pixel (arc detection remains sequential due to its nature)
    for i in range(n_pixels):
        bright_indices = find_longest_arc_optimized(bright_mask[i])
        dark_indices = find_longest_arc_optimized(dark_mask[i])

        # Select qualifying arc (preserve your exact logic)
        if len(bright_indices) >= required_n and len(bright_indices) >= len(dark_indices):
            arc_differences = differences[i, bright_indices]
        elif len(dark_indices) >= required_n:
            arc_differences = differences[i, dark_indices]
        else:
            continue

        # Calculate cornerness (preserve your exact options)
        match cornerness_calculation:
            case 'original':
                cornerness_scores[i] = np.sum(np.abs(arc_differences) - threshold)
            case 'sum_squared_diff':
                cornerness_scores[i] = np.sum(arc_differences ** 2)
            case 'mean_arc_diff':
                cornerness_scores[i] = np.mean(np.abs(arc_differences))

    return cornerness_scores


def fast_pipeline(image,
                             threshold_type='range_relative', threshold_factor=0.1,
                             circle_radius=3, n_ratio=0.56,
                             cornerness_calculation='original',
                             high_speed=True, visualize=False):
    """
    Vectorized FAST pipeline maintaining exact original interface and behavior

    Args: (identical to your original)
        image: Input grayscale image (numpy array)
        threshold_type: 'range_relative' or 'std_relative'
        threshold_factor: percentile for range_relative, integer for std_relative
        circle_radius: Bresenham circle radius
        n_ratio: arc ratio to qualify as a corner
        cornerness_calculation: 'original' or 'sum_squared_diff' or 'mean_arc_diff'
        high_speed: whether to use high speed test
        visualize: whether to generate visualization image
    Returns:
        List of tuples (x, y, cornerness)
    """

    # Validation and sanitization (preserve your logic)
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cache circle offsets for performance
    circle_offsets = generate_circle_offsets_cached(circle_radius)
    cardinal_indices = get_cardinal_indices(circle_offsets, circle_radius)

    # Calculate threshold (preserve your exact logic)
    match threshold_type:
        case 'range_relative':
            low, high = np.percentile(image, [10, 90])
            threshold = threshold_factor * (high - low)
        case 'std_relative':
            threshold = threshold_factor * np.std(image)
        case _:
            print(f"Unknown thresholding method:{threshold_type}")
            return [], None, None

    required_n = max(1, int(n_ratio * len(circle_offsets)))

    # Light denoising (preserve your preprocessing)
    image = cv2.bilateralFilter(image, 5, 15, 15)

    # Vectorized intensity extraction
    center_intensities, circle_intensities, y_coords, x_coords = \
        extract_circle_intensities_vectorized(image, circle_offsets, circle_radius)

    # Apply high speed test if enabled
    if high_speed:
        valid_mask = vectorized_high_speed_test(
            center_intensities, circle_intensities, cardinal_indices,
            threshold, n_ratio
        )
        center_intensities = center_intensities[valid_mask]
        circle_intensities = circle_intensities[valid_mask]
        y_flat_valid = y_coords.flatten()[valid_mask]
        x_flat_valid = x_coords.flatten()[valid_mask]
    else:
        y_flat_valid = y_coords.flatten()
        x_flat_valid = x_coords.flatten()

    if len(center_intensities) == 0:
        corner_list = []
    else:
        # Vectorized arc detection and cornerness
        cornerness_scores = vectorized_arc_and_corner(
            center_intensities, circle_intensities, threshold,
            required_n, cornerness_calculation
        )

        # Create strength map
        height, width = image.shape
        strength_map = np.zeros((height, width), dtype=np.float32)

        valid_corners = cornerness_scores > 0
        if valid_corners.any():
            strength_map[y_flat_valid[valid_corners], x_flat_valid[valid_corners]] = \
                cornerness_scores[valid_corners]

        # Peak detection (preserve your logic)
        peaks = peak_local_max(strength_map, min_distance=circle_radius)
        corner_list = [(x, y, strength_map[y, x]) for y, x in peaks]

    # Visualization (preserve your exact visualization logic)
    corner_img = None
    heatmap = None

    if visualize:
        corner_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        if corner_list:
            max_response = max(score for _, _, score in corner_list)
            for x, y, score in corner_list:
                radius = int(2 + (score / max_response) * 4)
                cv2.circle(corner_img, (x, y), radius, (255, 0, 0), 1)

        # Create response heatmap
        norm_response = cv2.normalize(strength_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)

    return corner_list, corner_img, heatmap


def process_single_parameter_set(param_set, distortion_no, ref_img_list):
    """Process a single parameter combination across all distortion levels."""
    threshold_method, threshold_factor, radius, n_ratio, cornerness_method = param_set

    # Compute reference corners once per parameter set
    ref_corners_dict = _compute_reference_corners(
        ref_img_list, threshold_method, threshold_factor, radius, n_ratio, cornerness_method
    )

    # Process all distortion levels for this parameter set
    results = []
    for level in range(1, 6):
        result = _process_distortion_level(
            distortion_no, level, ref_corners_dict,
            threshold_method, threshold_factor, radius, n_ratio, cornerness_method
        )
        results.append(result)

    return results


def _compute_reference_corners(ref_img_list, threshold_method, threshold_factor, radius, n_ratio, cornerness_method):
    """Compute corners for all reference images with given parameters."""
    ref_corners_dict = {}
    for img_no in ref_img_list:
        ref_image_path = f"tid2013/reference_images/I{img_no}.BMP"
        ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
        if ref_image is None:
            continue

        ref_corners, *_ = fast_pipeline(
            ref_image, threshold_type=threshold_method, threshold_factor=threshold_factor,
            circle_radius=radius, n_ratio=n_ratio, cornerness_calculation=cornerness_method
        )
        ref_corners_dict[img_no] = ref_corners
    return ref_corners_dict


def _process_distortion_level(distortion_no, level, ref_corners_dict, threshold_method,
                              threshold_factor, radius, n_ratio, cornerness_method):
    """Process a single distortion level for given parameters."""
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

        dist_corners, *_ = fast_pipeline(
            dist_image, threshold_type=threshold_method, threshold_factor=threshold_factor,
            circle_radius=radius, n_ratio=n_ratio, cornerness_calculation=cornerness_method
        )

        # Find matches with 3px threshold
        from harris_main import find_corner_matches
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
        'threshold_method': {'range_relative': 0, 'std_relative': 1}.get(threshold_method),
        'threshold_factor': float(threshold_factor), 'circle_radius': int(radius),
        'n_ratio': float(n_ratio),
        'cornerness_method': {'original': 0, 'sum_squared_diff': 1, 'mean_arc_diff': 2}.get(cornerness_method),
        'total_ref_corners': int(total_ref_corners), 'total_matches': int(total_matches),
        'sampling_num': int(total_images), 'mean_repeatability': float(mean_repeatability),
        'mean_localization': float(mean_localization),
        'mean_resp_ratio': float(mean_resp_ratio),
    }


def fast_wrapper(distortion_no: str, ref_img_list: List[str],
                 threshold_config_list: Dict[str, Tuple[float]] = None,
                 radius_list: Tuple = (3, 4, 5), n_ratio_list: List[int] = None,
                 cornerness_method_list: Tuple = ('original', 'sum_squared_diff', 'mean_arc_diff'),
                 n_jobs=-1):
    # Set defaults
    if threshold_config_list is None:
        threshold_config_list = {'range_relative': (0.04, 0.08), 'std_relative': (1.0, 2.0)}
    if n_ratio_list is None:
        n_ratio_list = [n / 16 for n in [7, 9, 10, 12, 14]]

    # Generate parameter combinations
    unpacked = [(k, v) for k, vals in threshold_config_list.items() for v in vals]
    param_combinations = list(itertools.product(unpacked, radius_list, n_ratio_list, cornerness_method_list))

    # Flatten parameter tuples for cleaner processing
    flattened_params = [
        (threshold_method, threshold_factor, radius, n_ratio, cornerness_method)
        for (threshold_method, threshold_factor), radius, n_ratio, cornerness_method in param_combinations
    ]

    print(f"Processing {len(flattened_params)} parameter sets for distortion #{distortion_no} "
          f"across {len(ref_img_list)} reference images using {n_jobs} jobs")

    # Parallel processing of parameter sets
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_parameter_set)(param_set, distortion_no, ref_img_list)
        for param_set in flattened_params
    )

    # Flatten results (each parameter set returns 5 distortion levels)
    flattened_results = [result for param_results in all_results for result in param_results]

    # Create and save dataframe
    df = pd.DataFrame(flattened_results)

    os.makedirs(f"results/distortion_{distortion_no}", exist_ok=True)
    output_filename = f"results/distortion_{distortion_no}/fast_metrics_dist{distortion_no}.parquet"
    df.to_parquet(output_filename)

    print(f"Data exported to {output_filename}")
    return df


if __name__ == "__main__":
    import time

    start_time = time.perf_counter()

    ref_img_list = ["01", "04", "08", "09", "13", "19"]
    _ = fast_wrapper("17", ref_img_list, n_jobs=14)

    end_time = time.perf_counter()
    print(f"=== Executed in {(end_time - start_time):.1f}s ===")

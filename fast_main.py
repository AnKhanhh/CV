import itertools
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import pandas as pd
from skimage.draw import circle_perimeter

import misc


# OpenCV's implementation: https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/fast.cpp#L56

def generate_circle_offsets(radius):
    """Generate circle offsets using skimage"""
    # (dx, dy) format, filter for unique values
    rr, cc = circle_perimeter(0, 0, radius)
    offsets = np.unique(np.column_stack([rr, cc]), axis=0).tolist()

    # Sort by angle for proper segment test ordering
    import math
    offsets.sort(key=lambda p: math.atan2(p[1], p[0]))
    return offsets


def calculate_threshold(image, threshold_type, threshold_factor):
    """Calculate adaptive threshold"""
    match threshold_type:
        case 'range_relative':
            low, high = np.percentile(image, [10, 90])
            return threshold_factor * (high - low)
        case 'std_relative':
            std_dev = np.std(image)
            return threshold_factor * std_dev
        case _:
            print(f"Unknown thresholding method:{threshold_type}")


def arc_and_corner(center_intensity, circle_intensities, threshold, required_n, cornerness_calculation):
    """Find qualifying arc from raw intensities and calculate cornerness"""
    circle_intensities = np.array(circle_intensities, dtype=np.float32)
    cornerness_calculation = cornerness_calculation.lower()

    # Create masks
    differences = circle_intensities - center_intensity
    bright_mask = differences > threshold
    dark_mask = differences < -threshold

    def get_longest_arc_indices(mask):
        """ duplicate, find arc, modulo indices back"""

        if not mask.any():
            return []
        n = len(mask)

        # Duplicate and concatenate
        doubled = np.concatenate([mask, mask])
        # Find where runs start and end (with padding to catch edges)
        padded = np.pad(doubled, 1, constant_values=False)
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            return []

        # Filter run (cap at mask length)
        lengths = ends - starts
        valid_runs = [(s, l) for s, l in zip(starts, lengths) if l <= n]

        # All True
        if not valid_runs:
            return list(range(n))

        # Get the longest run
        start, length = max(valid_runs, key=lambda x: x[1])

        # Modulo to handle index overshoot
        return [(start + i) % n for i in range(length)]

    bright_indices = get_longest_arc_indices(bright_mask)
    dark_indices = get_longest_arc_indices(dark_mask)

    # Select qualifying arc
    if len(bright_indices) >= required_n and len(bright_indices) >= len(dark_indices):
        arc_differences = differences[bright_indices]
    elif len(dark_indices) >= required_n:
        arc_differences = differences[dark_indices]
    else:
        return 0

    # Calculate cornerness from arc pixels only
    match cornerness_calculation:
        case 'original':
            return np.sum(np.abs(arc_differences) - threshold)
        case 'sum_squared_diff':
            return np.sum(arc_differences ** 2)
        case 'mean_arc_diff':
            return np.mean(np.abs(arc_differences))
        case _:
            return 0


def high_speed_test(center_intensity, cardinal_intensities, threshold, n_ratio):
    """A simple variation of the original high speed test"""
    cardinal_intensities = np.array(cardinal_intensities, dtype=np.float32)
    differences = cardinal_intensities - center_intensity
    bright_mask = differences > threshold
    dark_mask = differences < -threshold
    return max(bright_mask.sum(), dark_mask.sum()) >= (n_ratio // 0.25)


def fast_pipeline(image,
                  threshold_type='range_relative', threshold_factor=0.1,
                  circle_radius=3, n_ratio=0.56,
                  cornerness_calculation='original',
                  high_speed=True, visualize=False):
    """
    Core FAST detection implementation
    Args:
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
    from skimage.feature import peak_local_max

    # Validation and sanitization
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circle_offsets = generate_circle_offsets(circle_radius)
    cardinal_offsets = [(circle_radius, 0), (0, circle_radius), (-circle_radius, 0), (0, -circle_radius)]
    threshold = calculate_threshold(image, threshold_type, threshold_factor)
    required_n = max(1, int(n_ratio * len(circle_offsets)))
    corner_img = None
    heatmap = None

    # Lightly denoise
    image = cv2.bilateralFilter(image, 5, 15, 15)

    # Create strength map
    height, width = image.shape
    strength_map = np.zeros((height, width), dtype=np.float32)

    # Detect corners and fill strength map
    for y in range(circle_radius, height - circle_radius):
        for x in range(circle_radius, width - circle_radius):
            center_intensity = image[y, x]
            cardinal_intensities = [image[y + dy, x + dx] for dx, dy in cardinal_offsets]
            if high_speed and not high_speed_test(center_intensity, cardinal_intensities, threshold, n_ratio):
                continue
            circle_intensities = [image[y + dy, x + dx] for dx, dy in circle_offsets]
            cornerness = arc_and_corner(
                center_intensity, circle_intensities, threshold, required_n, cornerness_calculation
            )

            if cornerness > 0:
                strength_map[y, x] = cornerness

    peaks = peak_local_max(strength_map, min_distance=circle_radius)
    corner_list = [(x, y, strength_map[y, x]) for y, x in peaks]

    # points = np.float32([corner[:2] for corner in corner_list])
    # refined_coords = cv2.cornerSubPix(image, points, (5, 5), (-1, -1),
    #                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    # refined_corner_list = [(point[0], point[1], corner_list[i][2])
    #                        for i, point in enumerate(refined_coords)]

    if visualize:
        corner_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        max_response: float = max(score for _, _, score in corner_list) or 1.0
        for x, y, score in corner_list:
            radius = int(2 + (score / max_response) * 4)
            cv2.circle(corner_img, (x, y), radius, (255, 0, 0), 1)

        # Create response heatmap
        norm_response = cv2.normalize(strength_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)

    return corner_list, corner_img, heatmap


def fast_wrapper(distortion_no: str,
                 ref_img_list: List[str],
                 threshold_config_list: Dict[str, Tuple[float]] = None,
                 radius_list: Tuple = (3, 4, 5, 8),
                 n_ratio_list: List[int] = None,
                 cornerness_method_list: Tuple = ('original', 'sum_squared_diff', 'mean_arc_diff')):
    import math

    if threshold_config_list is None:
        threshold_config_list = {'range_relative': (0.04, 0.08, 0.12), 'std_relative': (1.0, 2.0, 3.0)}
    if n_ratio_list is None:
        n_list = [7, 9, 10, 12, 14]
        n_ratio_list = [(n / 16) for n in n_list]

    unpacked = [(k, v) for k, vals in threshold_config_list.items() for v in vals]
    total_iter = len(unpacked) * len(radius_list) * len(n_ratio_list) * len(cornerness_method_list)
    print(f"Processing {total_iter} parameter sets for distortion #{distortion_no} across {len(ref_img_list)} reference images")

    # Results list to build our dataframe
    results_df = []

    param_idx = 0
    for (threshold_method, threshold_factor), radius, n_ratio, cornerness_method in itertools.product(
        unpacked, radius_list, n_ratio_list, cornerness_method_list
    ):
        param_idx += 1
        misc.print_progress_bar(param_idx, total_iter)

        # Pre-calculate for all reference images as ground truth
        ref_corners_dict = {}
        for img_no in ref_img_list:
            ref_image_path = f"tid2013/reference_images/I{img_no}.BMP"
            ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
            if ref_image is None:
                print(f"Error loading {ref_image_path}")
                continue

            ref_corners, *_ = fast_pipeline(ref_image,
                                            threshold_type=threshold_method,
                                            threshold_factor=threshold_factor,
                                            circle_radius=radius,
                                            n_ratio=n_ratio,
                                            cornerness_calculation=cornerness_method)
            ref_corners_dict[img_no] = ref_corners

        # Process each distortion level
        for level in (1, 2, 3, 4, 5):
            # Initialize metrics
            total_ref_corners = 0
            total_matches = 0
            total_images = 0
            sum_repeatability = 0.0
            loc_distances = []
            resp_ratios = []
            resp_differences = []  # Store difference: reference - distorted

            # Process each reference image's distorted version
            for img_no, ref_corners in ref_corners_dict.items():
                # Load distorted image
                dist_image_path = f"tid2013/distorted_images/i{img_no}_{distortion_no}_{level}.bmp"
                dist_image = cv2.imread(dist_image_path, cv2.IMREAD_COLOR)
                if dist_image is None:
                    print(f"Error loading {dist_image_path}")
                    continue

                # Get corners for distorted image
                dist_corners, *_ = fast_pipeline(dist_image,
                                                 threshold_type=threshold_method,
                                                 threshold_factor=threshold_factor,
                                                 circle_radius=radius,
                                                 n_ratio=n_ratio,
                                                 cornerness_calculation=cornerness_method)

                # Find matches and calculate metrics
                from harris_main import find_corner_matches
                matches, cost_matrix = find_corner_matches(ref_corners, dist_corners)

                # Update metrics
                total_ref_corners += len(ref_corners)
                total_matches += len(matches)
                total_images += 1

                # Calculate repeatability for this image
                if len(ref_corners) > 0:
                    repeatability = len(matches) / len(ref_corners)
                    sum_repeatability += repeatability

                # Collect raw localization distances and response differences
                for i_ref, i_dist in matches:
                    # Add localization distance
                    loc_distance = cost_matrix[i_ref, i_dist]
                    loc_distances.append(float(loc_distance))

                    # Add response metrics
                    ref_resp = ref_corners[i_ref][2]
                    dist_resp = dist_corners[i_dist][2]
                    resp_ratios.append(float(dist_resp / ref_resp))  # Ratio: distorted/reference
                    resp_differences.append(float(ref_resp - dist_resp))  # Difference: reference - distorted

            # Calculate aggregate metrics
            mean_repeatability = 0.0
            if total_images > 0:
                mean_repeatability = sum_repeatability / total_images

            mean_localization = float('inf')
            if loc_distances:
                mean_localization = np.mean(loc_distances)

            mean_resp_ratio = float('nan')
            if resp_ratios:
                mean_resp_ratio = np.mean(resp_ratios)

            # Create result row for this parameter set and distortion level
            result_row = {
                'dt_no': int(distortion_no),
                'dt_lv': int(level),
                'threshold_method': {'range_relative': 0, 'std_relative': 1}.get(threshold_method),
                'threshold_factor': float(threshold_factor),
                'circle_radius': int(radius),
                'n_ratio': float(n_ratio),
                'cornerness_method': {'original': 0, 'sum_squared_diff': 1, 'mean_arc_diff': 2}.get(cornerness_method),
                'total_ref_corners': int(total_ref_corners),
                'total_matches': int(total_matches),
                'sampling_num': int(total_images),
                'mean_repeatability': float(mean_repeatability),
                'mean_localization': float(mean_localization),
                'loc_distances': loc_distances,  # Raw list of localization distances
                'mean_resp_ratio': float(mean_resp_ratio),
                'resp_ratios': resp_ratios,  # Raw list of response ratios
            }

            # Add to results
            results_df.append(result_row)

    # Convert to dataframe
    df = pd.DataFrame(results_df)

    # Convert Python lists to numpy arrays
    for col in ['loc_distances', 'resp_ratios']:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float32))

    # Save as Parquet
    os.makedirs(f"results/distortion_{distortion_no}", exist_ok=True)
    output_filename = f"results/distortion_{distortion_no}/fast_metrics_dist{distortion_no}.parquet"
    df.to_parquet(output_filename)

    print(f"Data exported to {output_filename}")

    return df

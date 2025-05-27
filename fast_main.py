import numpy as np
import cv2
from skimage.draw import circle_perimeter


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
    if threshold_type == 'range_relative':
        min_val, max_val = np.min(image), np.max(image)
        dynamic_range = max_val - min_val
        return threshold_factor * dynamic_range
    elif threshold_type == 'std_relative':
        std_dev = np.std(image)
        return threshold_factor * std_dev
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")


def arc_and_corner(center_intensity, circle_intensities, threshold, required_n, cornerness_calculation):
    """
    Find qualifying arc from raw intensities and calculate cornerness
    Returns: cornerness value or 0
    """
    circle_intensities = np.array(circle_intensities)
    cornerness_calculation = cornerness_calculation.lower()

    # Create masks
    differences = circle_intensities - center_intensity
    bright_mask = differences > threshold
    dark_mask = differences < -threshold

    # Find longest contiguous sequences
    def find_longest_contiguous(mask):
        if not np.any(mask):
            return 0
        # Duplicate to handle wrap-around
        extended = np.concatenate([mask, mask])
        max_length = 0
        current_length = 0
        for val in extended:
            if val:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        return min(max_length, len(mask))

    # Only one arc qualify since n > half_circle
    if find_longest_contiguous(bright_mask) >= required_n:
        qualifying_differences = differences[bright_mask]
    elif find_longest_contiguous(dark_mask) >= required_n:
        qualifying_differences = differences[dark_mask]
    else:
        return 0

    match cornerness_calculation:
        case 'max_arc_diff':  # Original method
            if np.mean(qualifying_differences) > 0:
                return np.sum(qualifying_differences - threshold)
            else:  # Dark arc
                return np.sum(-qualifying_differences - threshold)
        case 'sum_arc_diff':
            return np.sum(np.abs(qualifying_differences))
        case 'mean_arc_diff':
            return np.mean(np.abs(qualifying_differences))
        case _:
            print(f"Unknown cornerness calculation method: {cornerness_calculation}")


def fast_pipeline(image,
                  threshold_type, threshold_factor,
                  circle_radius=3, n_ratio=0.75,
                  cornerness_calculation='max_arc_diff',
                  nms=True, visualize=False):
    """
    Core FAST detection implementation
    Args:
        image: Input grayscale image (numpy array)
        threshold_type: 'range_relative' or 'std_relative'
        threshold_factor: percentile for range_relative, integer for std_relative
        circle_radius: Bresenham circle radius
        n_ratio: arc ratio to qualify as a corner
        cornerness_calculation: 'max_arc_diff' or 'sum_arc_diff' or 'mean_arc_diff'
        nms:whether to apply NMS

    Returns:
        List of tuples (x, y, cornerness)
    """
    from skimage.feature import peak_local_max

    # Validation and sanitization
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circle_offsets = generate_circle_offsets(circle_radius)
    threshold = calculate_threshold(image, threshold_type, threshold_factor)
    required_n = max(1, int(n_ratio * len(circle_offsets)))
    corner_img = None
    heatmap = None

    # Create strength map
    height, width = image.shape
    strength_map = np.zeros((height, width), dtype=np.float32)

    # Detect corners and fill strength map
    for y in range(circle_radius, height - circle_radius):
        for x in range(circle_radius, width - circle_radius):
            center_intensity = image[y, x]
            circle_intensities = [image[y + dy, x + dx] for dx, dy in circle_offsets]
            cornerness = arc_and_corner(
                center_intensity, circle_intensities, threshold, required_n, cornerness_calculation
            )

            if cornerness > 0:
                strength_map[y, x] = cornerness

    if nms:
        peaks = peak_local_max(strength_map, min_distance=10, threshold_abs=0)
        corner_list = [(x, y, strength_map[y, x]) for y, x in peaks]
    else:
        y_coords, x_coords = np.nonzero(strength_map)
        corner_list = [(x, y, strength_map[y, x]) for x, y in zip(x_coords, y_coords)]

    if visualize:
        corner_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        max_response: float = max(score for _, _, score in corner_list) or 1.0
        for x, y, score in corner_list:
            radius = int(3 + (score / max_response) * 10)
            cv2.circle(corner_img, (x, y), radius, (255, 0, 0), 1)

        # Create response heatmap
        norm_response = cv2.normalize(strength_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)

    return corner_list, corner_img, heatmap

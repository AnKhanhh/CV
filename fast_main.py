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


def fast_pipeline(image,
                  threshold_type='range_relative', threshold_factor=0.1,
                  circle_radius=3, n_ratio=0.56,
                  cornerness_calculation='original',
                  visualize=False):
    """
    Core FAST detection implementation
    Args:
        image: Input grayscale image (numpy array)
        threshold_type: 'range_relative' or 'std_relative'
        threshold_factor: percentile for range_relative, integer for std_relative
        circle_radius: Bresenham circle radius
        n_ratio: arc ratio to qualify as a corner
        cornerness_calculation: 'original' or 'sum_squared_diff' or 'mean_arc_diff'
        nms: whether to apply NMS
        visualize: whether to generate visualization image
    Returns:
        List of tuples (x, y, cornerness)
    """
    from skimage.feature import peak_local_max

    # Validation and sanitization
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circle_offsets = generate_circle_offsets(circle_radius)
    # cardinal_offsets = [(circle_radius, 0), (0, circle_radius), (-circle_radius, 0), (0, -circle_radius)]
    threshold = calculate_threshold(image, threshold_type, threshold_factor)
    required_n = max(1, int(n_ratio * len(circle_offsets)))
    corner_img = None
    heatmap = None

    # Lightly denoise
    image = cv2.bilateralFilter(image, 5, 25, 25)

    # Create strength map
    height, width = image.shape
    strength_map = np.zeros((height, width), dtype=np.float32)

    # Detect corners and fill strength map
    for y in range(circle_radius, height - circle_radius):
        for x in range(circle_radius, width - circle_radius):
            center_intensity = image[y, x]
            # cardinal_intensities = [image[y + dy, x + dx] for dx, dy in cardinal_offsets]

            circle_intensities = [image[y + dy, x + dx] for dx, dy in circle_offsets]
            cornerness = arc_and_corner(
                center_intensity, circle_intensities, threshold, required_n, cornerness_calculation
            )

            if cornerness > 0:
                strength_map[y, x] = cornerness

    peaks = peak_local_max(strength_map, min_distance=circle_radius)
    corner_list = [(x, y, strength_map[y, x]) for y, x in peaks]

    points = np.float32([corner[:2] for corner in corner_list])
    refined_coords = cv2.cornerSubPix(image, points, (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    refined_corner_list = [(point[0], point[1], corner_list[i][2])
                           for i, point in enumerate(refined_coords)]

    if visualize:
        corner_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        max_response: float = max(score for _, _, score in corner_list) or 1.0
        for x, y, score in corner_list:
            radius = int(2 + (score / max_response) * 4)
            cv2.circle(corner_img, (x, y), radius, (255, 0, 0), 1)

        # Create response heatmap
        norm_response = cv2.normalize(strength_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)

    return refined_corner_list, corner_img, heatmap

import numpy as np
import cv2
from skimage.draw import circle_perimeter


# OpenCV's implementation: https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/fast.cpp#L56

def generate_circle_offsets(radius):
    """Generate circle offsets using skimage"""
    rr, cc = circle_perimeter(0, 0, radius)
    offsets = list(zip(cc, rr))  # (dx, dy) format

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


def find_longest_contiguous_arc(binary_array):
    """Find longest contiguous sequence in circular array"""
    if not np.any(binary_array):
        return 0

    # Duplicate array to handle wrap-around
    extended = np.concatenate([binary_array, binary_array])

    max_length = 0
    current_length = 0

    for i in range(len(extended)):
        if extended[i]:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    # Limit to original array length
    return min(max_length, len(binary_array))


def calculate_corner_strength(center_intensity, circle_intensities, threshold,
                              qualifying_bright, qualifying_dark, strength_measure):
    """Calculate corner strength based on measure type"""
    circle_intensities = np.array(circle_intensities)

    if strength_measure == 'max_arc_diff':
        # Original FAST strength measure
        bright_strength = 0
        dark_strength = 0

        if np.any(qualifying_bright):
            bright_diffs = circle_intensities[qualifying_bright] - center_intensity - threshold
            bright_strength = np.sum(bright_diffs)

        if np.any(qualifying_dark):
            dark_diffs = center_intensity - circle_intensities[qualifying_dark] - threshold
            dark_strength = np.sum(dark_diffs)

        return max(bright_strength, dark_strength)

    elif strength_measure == 'sum_arc_diff':
        # Sum of all differences in qualifying arc
        qualifying_mask = qualifying_bright | qualifying_dark
        if not np.any(qualifying_mask):
            return 0
        all_diffs = np.abs(circle_intensities - center_intensity)
        return np.sum(all_diffs[qualifying_mask])

    elif strength_measure == 'mean_arc_diff':
        # Mean of differences in qualifying arc
        qualifying_mask = qualifying_bright | qualifying_dark
        if not np.any(qualifying_mask):
            return 0
        all_diffs = np.abs(circle_intensities - center_intensity)
        return np.mean(all_diffs[qualifying_mask])

    else:
        raise ValueError(f"Unknown strength measure: {strength_measure}")


def apply_nms(corners, nms_radius=10):
    """Apply non-maximum suppression with static radius"""
    if not corners:
        return corners

    # Sort by cornerness (descending)
    corners_sorted = sorted(corners, key=lambda c: c[2], reverse=True)
    suppressed = np.zeros(len(corners_sorted), dtype=bool)
    final_corners = []

    for i, (x1, y1, strength1) in enumerate(corners_sorted):
        if suppressed[i]:
            continue

        final_corners.append((x1, y1, strength1))

        # Suppress nearby corners
        for j in range(i + 1, len(corners_sorted)):
            if suppressed[j]:
                continue

            x2, y2, strength2 = corners_sorted[j]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if distance <= nms_radius:
                suppressed[j] = True

    return final_corners


def detect_fast_corners(image, threshold_type, threshold_factor, circle_radius,
                        n_ratio, strength_measure, apply_nms=True):
    """
    FAST corner detection implementation

    Args:
        image: Input grayscale image (numpy array)
        threshold_type: 'range_relative' or 'std_relative'
        threshold_factor: 0.04-0.12 for range_relative, 1.0-3.0 for std_relative
        circle_radius: 3, 4, 5, or 8
        n_ratio: 0.5-0.8 (percentage of circle pixels that must be contiguous)
        strength_measure: 'max_arc_diff', 'sum_arc_diff', or 'mean_arc_diff'
        apply_nms: Boolean, whether to apply non-maximum suppression

    Returns:
        List of tuples (x, y, cornerness)
    """
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate circle offsets
    circle_offsets = generate_circle_offsets(circle_radius)

    # Calculate adaptive threshold
    threshold = calculate_threshold(image, threshold_type, threshold_factor)

    # Calculate required contiguous pixels
    required_n = max(1, int(n_ratio * len(circle_offsets)))

    # Detect corners - exclude boundary pixels within radius distance
    corners = []
    height, width = image.shape

    for y in range(circle_radius, height - circle_radius):
        for x in range(circle_radius, width - circle_radius):
            center_intensity = image[y, x]
            circle_intensities = []

            # Sample circle pixels
            for dx, dy in circle_offsets:
                nx, ny = x + dx, y + dy
                circle_intensities.append(image[ny, nx])

            # Classify pixels
            bright_pixels = np.array(circle_intensities) > (center_intensity + threshold)
            dark_pixels = np.array(circle_intensities) < (center_intensity - threshold)

            # Find longest contiguous arcs
            bright_arc_length = find_longest_contiguous_arc(bright_pixels)
            dark_arc_length = find_longest_contiguous_arc(dark_pixels)

            # Check if either arc meets the requirement
            max_arc_length = max(bright_arc_length, dark_arc_length)
            is_corner = max_arc_length >= required_n

            if is_corner:
                # Calculate corner strength
                strength = calculate_corner_strength(
                    center_intensity, circle_intensities, threshold,
                    bright_pixels, dark_pixels, strength_measure
                )

                corners.append((x, y, strength))

    # Apply non-maximum suppression if enabled
    if apply_nms:
        corners = apply_nms(corners, nms_radius=10)

    return corners


# Example usage
if __name__ == "__main__":
    # Load test image
    image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Detect corners with specific parameters
    corners = detect_fast_corners(
        image=image,
        threshold_type='range_relative',
        threshold_factor=0.08,
        circle_radius=3,
        n_ratio=0.75,
        strength_measure='max_arc_diff',
        apply_nms=True
    )

    print(f"Detected {len(corners)} corners")
    print(f"Sample corners: {corners[:5]}")

    # Visualize results
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y, strength in corners:
        cv2.circle(result_image, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imshow('FAST Corners', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

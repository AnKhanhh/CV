import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional


def find_harris_corners(
    input_img: np.ndarray[np.uint8],
    k: float = 0.04,
    window_size: int = 5,
    threshold: float = 10000.00,
    window_type: str = 'gaussian',
    window_sigma: float = 1.5
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Detect Harris corners in the input grayscale image.
    Returns:
        List of detected corners as [x, y, response_value]
        Original image with detected corners marked in red
    """
    corner_list = []
    window_size = window_size | 1  # round up to odd number

    # Create a copy for visualization
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)

    # 1. Apply Gaussian blur, then calculate gradient
    # Smaller sigma value for more subtle blur
    blurred_img = cv2.GaussianBlur(input_img, (3, 3), 0.8)
    dx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=5)

    # 2. Compute products of derivatives
    i_xx = dx ** 2
    i_xy = dy * dx
    i_yy = dy ** 2

    height, width = input_img.shape

    # 3. Calculate Corner response and apply thresholding
    # Create the appropriate window kernel based on window_type
    if window_type.lower() == 'gaussian':
        kernel_1d = cv2.getGaussianKernel(window_size, window_sigma)
    elif window_type.lower() == 'binary':
        kernel_1d = np.ones(window_size, dtype=np.float32) / window_size
    else:
        print('Err: unrecognized window type in Harris function')
        return [], output_img

    # Apply the window using separable filtering (more efficient)
    sum_i_xx = cv2.sepFilter2D(i_xx, -1, kernel_1d, kernel_1d, borderType=cv2.BORDER_REFLECT_101)
    sum_i_xy = cv2.sepFilter2D(i_xy, -1, kernel_1d, kernel_1d, borderType=cv2.BORDER_REFLECT_101)
    sum_i_yy = cv2.sepFilter2D(i_yy, -1, kernel_1d, kernel_1d, borderType=cv2.BORDER_REFLECT_101)

    # Compute Harris response
    det = (sum_i_xx * sum_i_yy) - (sum_i_xy ** 2)
    trace = sum_i_xx + sum_i_yy
    r = det - k * (trace ** 2)

    # Find corners above threshold
    for y in range(height):
        for x in range(width):
            if r[y, x] > threshold:
                corner_list.append((x, y, r[y, x]))
                output_img[y, x] = (0, 0, 255)  # Mark corner in red

    return corner_list, output_img


if __name__ == "__main__":
    input = cv2.imread("tid2013/reference_images/I01.BMP", cv2.IMREAD_GRAYSCALE)
    print(input.shape)
    plt.imshow(input, cmap='gray')
    pass

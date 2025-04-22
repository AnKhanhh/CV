import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional


# Opencv's implementation
# https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/corner.cpp#L634
def harris_optimized(
    input_img: np.ndarray,
    k: float = 0.04,
    neighborhood_size: int = 5,
    aperture_size: int = 3,
    threshold_ratio: float = 0.01,
    nms_radius: int = 5,
    border_type: int = cv2.BORDER_REFLECT_101,
    window_type: str = 'gaussian'
) -> Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray]:
    """
    Detect Harris corners in the input grayscale image
    Returns:
        List of detected corners as (x, y, response_value)
        Original image with detected corners marked in red
        Heatmap of Harris corner response
    """
    try:
        # 1. Param validation
        neighborhood_size = neighborhood_size | 1  # always odd
        aperture_size = aperture_size | 1  # always odd
        window_type = window_type.lower()  # always lowercase
        # Create copy for visualization
        if len(input_img.shape) == 2:
            output_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        else:
            output_img = input_img.copy()
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # Enforce depth
        if input_img.dtype != np.float32:
            input_img = input_img.astype(np.float32)

        # 2. Calculate gradients and products
        dx = cv2.Sobel(input_img, cv2.CV_32F, 1, 0, ksize=aperture_size, borderType=border_type)
        dy = cv2.Sobel(input_img, cv2.CV_32F, 0, 1, ksize=aperture_size, borderType=border_type)
        i_xx = dx * dx
        i_xy = dx * dy
        i_yy = dy * dy

        # 3. Sum the products of gradients
        if window_type == 'gaussian':
            kernel_1d = cv2.getGaussianKernel(neighborhood_size, 0)  # sigma = 0.3*((ksize-1)*0.5 – 1) + 0.8
        elif window_type == 'binary' or window_type == 'uniform':
            kernel_1d = np.ones(neighborhood_size) / neighborhood_size
        else:
            raise Exception("unknown window type")
        sum_i_xx = cv2.sepFilter2D(i_xx, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)
        sum_i_xy = cv2.sepFilter2D(i_xy, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)
        sum_i_yy = cv2.sepFilter2D(i_yy, cv2.CV_32F, kernel_1d, kernel_1d, borderType=border_type)

        # 4.Calculate Harris response
        det = (sum_i_xx * sum_i_yy) - (sum_i_xy * sum_i_xy)
        trace = sum_i_xx + sum_i_yy
        harris_response = det - k * (trace * trace)

        # 5. Apply NMS, then relative thresholding based on max value
        coordinates: Tuple[List[int], List[int]] = peak_local_max(harris_response, min_distance=nms_radius, threshold_rel=threshold_ratio)

        # 6. Post-processing steps
        # Create list of corners
        corner_list: List[Tuple[int, int, float]] = [(x, y, harris_response[y, x]) for y, x in coordinates]

        # Mark corner, radius based on response strength
        max_response: float = max(score for _, _, score in corner_list) or 1.0
        for x, y, score in corner_list:
            radius = int(3 + (score / max_response) * 10)
            cv2.circle(output_img, (x, y), radius, (0, 0, 255), 1)

        # Create response heatmap
        norm_response = cv2.normalize(harris_response, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_response.astype(np.uint8), cv2.COLORMAP_JET)

        return corner_list, output_img, heatmap

    except Exception as e:
        print(f"Error in Harris corner detection: {e}")
        return [], input_img, np.zeros((*input_img.shape[:2], 3), dtype=np.float32)


if __name__ == "__main__":
    input = cv2.imread("tid2013/reference_images/I01.BMP", cv2.IMREAD_GRAYSCALE)
    print(input.shape)
    plt.imshow(input, cmap='gray')
    pass

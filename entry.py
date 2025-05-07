import cv2
from matplotlib import pyplot as plt

from harris_main import cv2_harris_wrapper


def test_cv2_reference_image():
    image = cv2.imread("tid2013/reference_images/I01.BMP", cv2.IMREAD_GRAYSCALE)
    # Detect corners
    corners, visualization = cv2_harris_wrapper(
        image, block_size=2, aperture_size=3,
        k=0.04, threshold_ratio=0.01,
        min_distance=10,
        visualize=True
    )


test_cv2_reference_image()

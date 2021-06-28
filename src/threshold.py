import cv2
import numpy as np


class Threshold:
    @staticmethod
    def filter_bgr(img, bgr_thresh=[(0, 0, 0), (255, 255, 255)], **kwargs):
        src = np.copy(img)
        return cv2.inRange(src, np.uint8(bgr_thresh[0]), np.uint8(bgr_thresh[1]))

    @staticmethod
    def filter_hsv(img, hsv_thresh=[(0, 0, 0), (255, 255, 255)], **kwargs):
        src = np.copy(img)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        return cv2.inRange(src, np.uint8(hsv_thresh[0]), np.uint8(hsv_thresh[1]))

    @staticmethod
    def filter_hls(img, hls_thresh=[(0, 0, 0), (255, 255, 255)], **kwargs):
        src = np.copy(img)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)
        return cv2.inRange(src, np.uint8(hls_thresh[0]), np.uint8(hls_thresh[1]))

    @staticmethod
    def filter_lab(img, lab_thresh=[(0, 0, 0), (255, 255, 255)], **kwargs):
        src = np.copy(img)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        return cv2.inRange(src, np.uint8(lab_thresh[0]), np.uint8(lab_thresh[1]))

    @staticmethod
    def sobel(sobel_result, thresh=(0, 255), **kwargs):
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel_result)

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude
        #    is > thresh_min and < thresh_max
        binary_sobel = np.zeros_like(scaled_sobel, dtype=np.uint8)
        binary_sobel[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_sobel

    @staticmethod
    def magnitude(sobel_x, sobel_y, thresh=(0, 255), **kwargs):
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))

        # Create a binary mask where mag thresholds are met
        binary_magnitude = np.zeros_like(scaled_sobel, dtype=np.uint8)
        binary_magnitude[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_magnitude

    @staticmethod
    def direction(sobel_x, sobel_y, thresh=(0, np.pi / 2), **kwargs):
        # Take the absolute value of the x and y gradients
        abs_sobel_x = np.absolute(sobel_x)
        abs_sobel_y = np.absolute(sobel_y)

        # Use np.arctan2(abs_sobel_y, abs_sobel_x) to calculate the direction of the gradient
        grad_direction = np.arctan2(abs_sobel_y, abs_sobel_x)

        # Create a binary mask where direction thresholds are met
        binary_direction = np.zeros_like(grad_direction, dtype=np.uint8)
        binary_direction[
            (grad_direction >= thresh[0]) & (grad_direction <= thresh[1])
        ] = 1
        return binary_direction

    @staticmethod
    def gradient(
        img,
        sobel_kernel=15,
        sobel_thresh=(20, 200),
        magnitude_thresh=(30, 200),
        direction_thresh=(0.2, 1.3),
        **kwargs,
    ):
        # 1 - Convert to grayscale
        img_gray = np.copy(img[:, :, 2])

        # Get Sobel derivatives in both X & Y directions
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Get Sobel threshold gradients
        bin_sobel_x = Threshold.sobel(sobel_x, thresh=sobel_thresh)
        # bin_sobel_y = Threshold.sobel(sobel_y, thresh=sobel_thresh)

        # Get Sobel magnitude gradients
        bin_magnitude = Threshold.magnitude(sobel_x, sobel_y, thresh=magnitude_thresh)

        # Get Sobel directional gradients
        bin_direction = Threshold.direction(sobel_x, sobel_y, thresh=direction_thresh)

        return ((bin_sobel_x) | (bin_magnitude & bin_direction)) * 255

    @staticmethod
    def combined_threshold(img, **kwargs):
        """
        Applies Gradient & Color threshold to the images
        Args:
            img: Image to apply threshold on
            sobel_kernel: Sobel Kernel size
            sobel_thresh: (min, max) values to filter Sobel derivative
            magnitude_thresh: (min, max) values to filter Sobel derivative's magnitude
            direction_thresh: (min, max) values to filter Sobel derivative's direction
            hsv_thresh: (min, max) values to filter in HSV color space
            hls_thresh: (min, max) values to filter in HLS color space
            lab_thresh: (min, max) values to filter in LAB color space

        Example:
            NOTE: BEST CONFIG for both normal & hard videos
            hsv = Threshold.filter_hsv(img, hsv_thresh=[(0, 0, 150), (179, 255, 255)])
            hls = Threshold.filter_hls(img, hls_thresh=[(0, 160, 0), (179, 255, 255)])
            lab = Threshold.filter_lab(img, lab_thresh=[(0, 0, 180), (255, 255, 255)])

            NOTE: Best for third video
            hsv = Threshold.filter_hsv(img, hsv_thresh=[(0, 0, 0), (179, 41, 255)])
            hls = Threshold.filter_hls(img, hls_thresh=[(0, 0, 0), (179, 30, 25)])
            lab = Threshold.filter_lab(img, lab_thresh=[(0, 0, 150), (255, 255, 255)])
        """
        grad = Threshold.gradient(img, **kwargs)
        hls = Threshold.filter_hls(img, **kwargs)
        hsv = Threshold.filter_hsv(img, **kwargs)
        lab = Threshold.filter_lab(img, **kwargs)
        return grad & (hls | hsv | lab)

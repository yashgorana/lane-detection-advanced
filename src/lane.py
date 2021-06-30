from src.measurements import Measurements
import numpy as np


class LaneLine:
    # TODO:

    def __init__(self, pixels):
        self.px_x, self.px_y = pixels

        if len(self.px_x) == 0 or len(self.px_y) == 0:
            self.self.fit = None
        else:
            self.fit = np.polyfit(self.px_x, self.px_y, 2)


class FindLanes:
    LANE_HISTORY_WEIGHTS = (1, 0.75, 0.25, 0.25, 0.25, 0.2, 0.2, 0.125, 0.125, 0.1, 0.1)

    @staticmethod
    def fit_poly(img_shape, leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    @staticmethod
    def clipped_window(x, y, w, h, max_x, max_y):
        x1 = np.clip(x, 0, max_x)
        y1 = np.clip(y, 0, max_y)
        x2 = np.clip(x + w, 0, max_x)
        y2 = np.clip(y + h, 0, max_y)
        return (x1, y1), (x2, y2)

    @staticmethod
    def intersect_with_window(window, arr):
        (x1, y1), (x2, y2) = window
        (x, y) = arr
        return (y >= y1) & (y < y2) & (x >= x1) & (x < x2)

    @staticmethod
    def focused_search(img_binary, previous_lane, search_width=20, **kwargs):
        left_fit, right_fit = previous_lane

        # Grab activated pixels
        nonzero = img_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin_arr = np.array([0, 0, search_width])
        left_search_region = np.polyval(left_fit - margin_arr, nonzeroy), np.polyval(
            left_fit + margin_arr, nonzeroy
        )
        right_search_region = np.polyval(right_fit - margin_arr, nonzeroy), np.polyval(
            right_fit + margin_arr, nonzeroy
        )

        left_lane_inds = (nonzerox > left_search_region[0]) & (
            nonzerox < left_search_region[1]
        )
        right_lane_inds = (nonzerox > right_search_region[0]) & (
            nonzerox < right_search_region[1]
        )

        # Again, extract left and right line pixel positions
        left_x = nonzerox[left_lane_inds]
        left_y = nonzeroy[left_lane_inds]
        right_x = nonzerox[right_lane_inds]
        right_y = nonzeroy[right_lane_inds]

        return {
            "type": "focused_search",
            "lane_pixels": (left_x, left_y, right_x, right_y),
            "region_points": (left_search_region, right_search_region),
            "search_width": search_width,
        }

    @staticmethod
    def histogram_search(
        img_binary, sw_count=10, sw_width=160, sw_minpix=800, **kwargs
    ):
        im_h, im_w = img_binary.shape

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img_binary[im_h // 2 :, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.uint16(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        sw_height = np.uint16(im_h // sw_count)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        sliding_windows = []

        # Step through the windows one by one
        for window in range(sw_count):
            # Identify window boundaries in x and y (and right and left)
            win_y_top = im_h - (window + 1) * sw_height

            left_window = FindLanes.clipped_window(
                x=(leftx_current - int(sw_width / 2)),
                y=win_y_top,
                w=sw_width,
                h=sw_height,
                max_x=im_w,
                max_y=im_h,
            )
            right_window = FindLanes.clipped_window(
                x=(rightx_current - int(sw_width / 2)),
                y=win_y_top,
                w=sw_width,
                h=sw_height,
                max_x=im_w,
                max_y=im_h,
            )
            sliding_windows.append((left_window, right_window))

            good_left_inds = FindLanes.intersect_with_window(
                window=left_window, arr=(nonzero_x, nonzero_y)
            )
            good_right_inds = FindLanes.intersect_with_window(
                window=right_window, arr=(nonzero_x, nonzero_y)
            )

            num_good_left = good_left_inds.nonzero()[0]
            num_good_right = good_right_inds.nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(num_good_left)
            right_lane_inds.append(num_good_right)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(num_good_left) > sw_minpix:
                leftx_current = np.uint16(np.mean(nonzero_x[good_left_inds]))
            if len(num_good_right) > sw_minpix:
                rightx_current = np.uint16(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        return {
            "type": "histogram_search",
            "lane_pixels": (left_x, left_y, right_x, right_y),
            "sliding_windows": sliding_windows,
            "histogram": histogram,
        }

    @staticmethod
    def find_lane_lines(img_binary, **kwargs):
        """Searches for lane lines. Checks for lane sanity. Averages out from history"""
        im_h, im_w = img_binary.shape
        lane_history = kwargs.get("lane_history", [])
        previous_lane = lane_history[-1] if len(lane_history) > 0 else None

        # Find our lane pixels
        if previous_lane is None:
            search_result = FindLanes.histogram_search(img_binary, **kwargs)
        else:
            search_result = FindLanes.focused_search(
                img_binary, previous_lane.get("fit"), **kwargs
            )

        left_x, left_y, right_x, right_y = search_result.get("lane_pixels")

        if (
            len(left_x) == 0
            or len(left_y) == 0
            or len(right_x) == 0
            or len(right_y) == 0
        ):
            return previous_lane

        # Fit a second order polynomial to each using `np.polyfit`
        left_lane_fit = np.polyfit(left_y, left_x, 2)
        right_lane_fit = np.polyfit(right_y, right_x, 2)

        # Weighted average of lane history
        if len(lane_history) > 0:
            prev_left_lane_fit = []
            prev_right_lane_fit = []

            for history in lane_history:
                p_left, p_right = history.get("fit")
                prev_left_lane_fit.append(p_left)
                prev_right_lane_fit.append(p_right)

            left_all = [left_lane_fit] + prev_left_lane_fit
            right_all = [right_lane_fit] + prev_right_lane_fit

            left_lane_fit = np.average(
                left_all,
                axis=0,
                weights=FindLanes.LANE_HISTORY_WEIGHTS[: len(left_all)],
            )
            right_lane_fit = np.average(
                right_all,
                axis=0,
                weights=FindLanes.LANE_HISTORY_WEIGHTS[: len(right_all)],
            )

        plot_y = np.linspace(0, im_h - 1, im_h)
        left_lane_pts = np.int32([np.polyval(left_lane_fit, plot_y), plot_y]).T
        right_lane_pts = np.int32([np.polyval(right_lane_fit, plot_y), plot_y]).T

        # TODO: Check lane sanity here

        roc = Measurements.measure_curvature(
            fit=(left_lane_fit, right_lane_fit), **kwargs
        )

        return {
            "fit": (left_lane_fit, right_lane_fit),
            "fit_points": (left_lane_pts, right_lane_pts),
            "pixels": search_result.get("lane_pixels"),
            "curvature_m": roc.get("radius"),
            "offset_m": roc.get("offset"),
            "search_result": search_result,
        }

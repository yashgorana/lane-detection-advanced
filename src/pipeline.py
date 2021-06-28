import cv2
import numpy as np

from .camera import Camera
from .lane import FindLanes
from .measurements import Measurements
from .threshold import Threshold
from .warp import Warp


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


class LaneDetection:
    def __init__(self):
        self._state = {}

    def _push_lane_history(self, lane_detection_result, max_results=5):
        lane_history = self._state.get("lane_history")
        if lane_history is None:
            self._state["lane_history"] = []

        self._state["lane_history"].append(lane_detection_result)
        self._state["lane_history"] = self._state["lane_history"][-max_results:]

    def detect(self, frame, **kwargs):

        result = np.copy(frame)

        # Undistort
        # => Returns undistorted image
        result = Camera.undistort(result, **kwargs)
        self._state["undistorted"] = result

        # Threshold
        # => Returns bionary image
        result = Threshold.combined_threshold(result, **kwargs)
        self._state["thresholded"] = result

        # Warp image
        # => Returns image & Warp matrices
        tx_s, tx_d = Warp.get_default_warp_points()
        result, M, M_inv = Warp.warp_image(result, tx_s, tx_d, **kwargs)
        self._state["tx_s"] = tx_s
        self._state["birds_eye"] = result

        # Find Lanes
        # => Returns left & right lane polyfit
        lane_history = self._state.get("lane_history", [])
        result = FindLanes.find_lane_lines(result, lane_history=lane_history, **kwargs)
        self._push_lane_history(result)

        # Fill Lanes
        # => Returns image with filled lanes in warped view
        result = self.draw_lane(frame.shape, result, **kwargs)
        self._state["filled_lane"] = result

        # Unwrap Lanes
        # => Returns the unwarped image with filled lanes
        result = Warp.unwarp_image(result, M_inv, **kwargs)
        self._state["final_result"] = result

        return {
            "result": result,
            "states": self._state,
        }

    def detect_and_render(
        self,
        frame,
        draw_metrics=True,
        draw_gradient=False,
        draw_lane_search=False,
        **kwargs,
    ):
        combine_with_original = True

        # Run lane detection
        lane_detection_result = self.detect(frame, **kwargs)
        # Extract result & undistorted source image
        result_img = lane_detection_result["result"]
        undistorted_src = np.copy(lane_detection_result["states"]["undistorted"])

        # Draw gradient lines
        if draw_gradient:
            grad = lane_detection_result["states"]["thresholded"]
            grad = np.dstack([grad, grad, grad])
            undistorted_src = cv2.addWeighted(undistorted_src, 1, grad, 0.8, 0)

        # If we need to draw search lines
        if not draw_gradient and draw_lane_search:
            birds_eye = lane_detection_result["states"]["birds_eye"]
            last_detected_lane = lane_detection_result["states"]["lane_history"][-1]
            search_type = last_detected_lane["search_result"]["type"]

            # Rendering search will replace the original frame with birds eye view
            combine_with_original = False

            if last_detected_lane and search_type == "histogram_search":
                undistorted_src = self.draw_search_histogram(
                    birds_eye, last_detected_lane
                )
            elif last_detected_lane and search_type == "focused_search":
                undistorted_src = self.draw_search_region(birds_eye, last_detected_lane)

        # Merge
        if combine_with_original:
            undistorted_src = cv2.addWeighted(undistorted_src, 1, result_img, 0.25, 0)

        # draw radius of curvature & offset
        if draw_metrics:
            self.draw_metrics(undistorted_src, lane_detection_result)

        return undistorted_src, frame

    def draw_lane(self, img_shape, lane_detection_result, **kwargs):
        frame = np.zeros(shape=img_shape).astype(np.uint8)
        if lane_detection_result is None:
            return frame
        # (left_lane_fit, right_lane_fit) = lane_detection_result.get('fit')
        (left_lane_pts, right_lane_pts) = lane_detection_result.get("fit_points")

        pts = np.hstack(([left_lane_pts], [np.flipud(right_lane_pts)]))

        # # Draw the lane onto the warped blank image
        cv2.fillPoly(frame, np.int_([pts]), (0, 255, 0))

        # Draw
        cv2.polylines(frame, [left_lane_pts], False, (0, 0, 255), 20)
        cv2.polylines(frame, [right_lane_pts], False, (0, 0, 255), 20)

        return frame

    def draw_search_histogram(self, birds_eye, lane_detection_result, **kwargs):
        result = np.copy(birds_eye)
        result = np.dstack([result, result, result])

        (left_x, left_y, right_x, right_y) = lane_detection_result["pixels"]
        (left_lane_pts, right_lane_pts) = lane_detection_result["fit_points"]
        sliding_windows = lane_detection_result["search_result"]["sliding_windows"]

        # Draw Sliding Windows
        for left_window, right_window in sliding_windows:
            cv2.rectangle(result, left_window[0], left_window[1], (0, 255, 0), 2)
            cv2.rectangle(result, right_window[0], right_window[1], (0, 255, 0), 2)

        # Highlight selected points
        result[left_y, left_x] = [255, 0, 0]
        result[right_y, right_x] = [0, 0, 255]

        # Draw lanes
        cv2.polylines(result, [left_lane_pts], False, (0, 255, 255), 10)
        cv2.polylines(result, [right_lane_pts], False, (0, 255, 255), 10)

        return result

    def draw_search_region(self, birds_eye, lane_detection_result):
        result = np.copy(birds_eye)
        result = np.dstack([result, result, result])
        overlay = np.zeros_like(result)

        (left_lane_pts, right_lane_pts) = lane_detection_result["fit_points"]
        (left_x, left_y, right_x, right_y) = lane_detection_result["pixels"]
        margin_arr = np.array([lane_detection_result["search_result"]["margin"], 0])

        left_lane_region = np.hstack(
            ([left_lane_pts - margin_arr], [np.flipud(left_lane_pts + margin_arr)])
        )
        right_lane_region = np.hstack(
            ([right_lane_pts - margin_arr], [np.flipud(right_lane_pts + margin_arr)])
        )

        # Highlight selected points
        result[left_y, left_x] = [255, 0, 0]
        result[right_y, right_x] = [0, 0, 255]

        # Draw search region
        cv2.fillPoly(overlay, np.int64([left_lane_region]), (0, 255, 0))
        cv2.fillPoly(overlay, np.int64([right_lane_region]), (0, 255, 0))

        result = cv2.addWeighted(result, 1, overlay, 0.3, 0)

        # Draw lanes
        cv2.polylines(result, [left_lane_pts], False, (0, 255, 255), 10)
        cv2.polylines(result, [right_lane_pts], False, (0, 255, 255), 10)

        return result

    def draw_metrics(self, img, lane_detection_result):
        last_detected_lane = lane_detection_result["states"]["lane_history"][-1]
        curvature_m = last_detected_lane.get("curvature_m", "0.0")
        offset_m = last_detected_lane.get("offset_m", "0.0")

        curvature_m = (
            f"Radius of curvature: L={curvature_m[1]:.2f}m, R={curvature_m[1]:.2f}m"
        )
        offset_m = f"Offset from center: {offset_m:.2f}m"

        cv2.putText(
            img,
            curvature_m,
            (50, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            offset_m,
            (50, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return img

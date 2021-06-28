import pickle
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Camera", "CALIBRATION_DATA_PATH"]

CALIBRATION_DATA_PATH = "camera_cal/calibration.dat"


class Camera:
    @staticmethod
    def undistort(distorted_img, calibration_data, **kwargs):
        (_, mtx, dist, _, _) = calibration_data
        return cv2.undistort(distorted_img, mtx, dist, None, mtx)

    @staticmethod
    def calibrate(img_paths, grid=(9, 6)):
        nx = grid[0]
        ny = grid[1]
        grid_indices = np.zeros((nx * ny, 3), np.float32)
        grid_indices[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        grid_list = []  # 3d points in real world space
        corners_list = []  # 2d points in image plane.

        img_shape = None

        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if not img_shape:
                img_shape = gray.shape[::-1]

            # Find the chessboard corners
            ret, detected_corners = cv2.findChessboardCorners(gray, grid, None)

            # If found, add object points, image points
            if ret == False:
                continue

            grid_list.append(grid_indices)
            corners_list.append(detected_corners)

        return cv2.calibrateCamera(grid_list, corners_list, img_shape, None, None)

    @staticmethod
    def save_calibration(path, caliberation_data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode="wb") as fp:
            pickle.dump(caliberation_data, fp)

    @staticmethod
    def load_calibration(path):
        path = Path(path)
        with path.open(mode="rb") as fp:
            return pickle.load(fp)

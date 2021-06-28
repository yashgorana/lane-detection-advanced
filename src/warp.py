import cv2
import numpy as np


class Warp:
    @staticmethod
    def warp_image(img, tx_src, tx_dest, **kwargs):
        img_size = (img.shape[1], img.shape[0])
        tx_src = tx_src.astype(np.float32)
        tx_dest = tx_dest.astype(np.float32)

        # Calculate the transformation matrix and it's inverse transformation
        M = cv2.getPerspectiveTransform(tx_src, tx_dest)
        M_inv = cv2.getPerspectiveTransform(tx_dest, tx_src)
        return cv2.warpPerspective(img, M, img_size, cv2.INTER_LINEAR), M, M_inv

    @staticmethod
    def unwarp_image(img, M_inv, **kwargs):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, M_inv, img_size, cv2.INTER_LINEAR)

    @staticmethod
    def get_default_warp_points():
        """Handpicked points warp transform source & destination points"""
        tx_src = np.int32([[260, 670], [570, 460], [720, 460], [1045, 670]])
        tx_dst = np.int32([[200, 680], [200, 000], [1000, 00], [1000, 680]])
        return (tx_src, tx_dst)

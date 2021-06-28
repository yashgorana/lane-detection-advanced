import numpy as np


class Measurements:
    @staticmethod
    def scale_fit(fit, m_per_pixel):
        # fit = a(y**2) + by + c
        # scaled_fit = a(y**2)(mx/(my**2)) + by(mx/my) + c(mx)
        xm, ym = m_per_pixel
        scaler = np.array([(xm / (ym ** 2)), (xm / ym), xm])
        return fit * scaler

    @staticmethod
    def measure_curvature(
        fit, m_per_pixel=(3.7 / 700, 30 / 720), im_shape=(1280, 720), **kwargs
    ):
        left_fit, right_fit = fit
        xm, ym = m_per_pixel
        x_eval, y_eval = im_shape

        # Convert eval values from px to m
        x_eval, y_eval = (x_eval * xm, y_eval * ym)

        left_fit_scaled = Measurements.scale_fit(left_fit, m_per_pixel)
        right_fit_scaled = Measurements.scale_fit(right_fit, m_per_pixel)

        eval_roc = lambda A, B, C, y: (1 + (2 * A * y + B) ** 2) ** (3 / 2) / abs(2 * A)

        # Measure radius of curvature
        rad_left = eval_roc(*left_fit_scaled, y_eval)
        rad_right = eval_roc(*right_fit_scaled, y_eval)

        # Evaluate scaled polyfit with scaled variable
        left_point = np.polyval(left_fit_scaled, y_eval)
        right_point = np.polyval(right_fit_scaled, y_eval)

        # Center is the average of the left & right points
        lane_center = (left_point + right_point) / 2
        image_center = x_eval / 2

        return {"radius": (rad_left, rad_right), "offset": lane_center - image_center}

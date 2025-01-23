import numpy as np


class GaussC10:
    """
    Provides the ten Wen and Breazeale coefficients for a multi-Gaussian beam model.
    """

    @staticmethod
    def get_coefficients():
        """
        Returns the coefficients a and b as two numpy arrays.

        Returns:
            tuple: (a, b) where:
                - a (numpy.ndarray): Array of complex coefficients 'a'.
                - b (numpy.ndarray): Array of complex coefficients 'b'.
        """
        a = np.array([
            11.428 + 0.95175j,
            0.06002 - 0.08013j,
            -4.2743 - 8.5562j,
            1.6576 + 2.7015j,
            -5.0418 + 3.2488j,
            1.1227 - 0.68854j,
            -1.0106 - 0.26955j,
            -2.5974 + 3.2202j,
            -0.14840 - 0.31193j,
            -0.20850 - 0.23851j,
        ])

        b = np.array([
            4.0697 + 0.22726j,
            1.1531 - 20.933j,
            4.4608 + 5.1268j,
            4.3521 + 14.997j,
            4.5443 + 10.003j,
            3.8478 + 20.078j,
            2.5280 - 10.310j,
            3.3197 - 4.8008j,
            1.9002 - 15.820j,
            2.6340 + 25.009j,
        ])

        return a, b

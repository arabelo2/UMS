import numpy as np


class DelayLaws2D:
    """
    Computes the time delay for an array transducer to steer and focus a beam.
    """

    @staticmethod
    def compute_delays(M, s, Phi, F, c):
        """
        Compute time delays for steering and focusing a beam.

        Parameters:
            M (int): Number of elements in the array.
            s (float): Pitch (distance between elements) in mm.
            Phi (float): Beam steering angle in degrees.
            F (float): Focal length in mm (use np.inf for steering only).
            c (float): Wave speed in m/s.

        Returns:
            numpy.ndarray: Array of time delays (in microseconds).
        """
        Mb = (M - 1) / 2
        m = np.arange(1, M + 1)  # Element indices
        em = s * ((m - 1) - Mb)  # Element centroids

        if np.isinf(F):  # Steering only
            if Phi > 0:
                td = 1000 * s * np.sin(np.radians(Phi)) * (m - 1) / c
            else:
                td = 1000 * s * np.sin(np.radians(abs(Phi))) * (M - m) / c
        else:  # Steering and focusing
            r1 = np.sqrt(F**2 + (Mb * s)**2 + 2 * F * Mb * s * np.sin(np.radians(Phi)))
            rm = np.sqrt(F**2 + em**2 - 2 * F * em * np.sin(np.radians(Phi)))
            rM = np.sqrt(F**2 + (Mb * s)**2 + 2 * F * Mb * s * np.sin(np.radians(abs(Phi))))
            if Phi > 0:
                td = 1000 * (r1 - rm) / c
            else:
                td = 1000 * (rM - rm) / c

        return td

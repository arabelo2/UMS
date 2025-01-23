import numpy as np


class Elements:
    """
    Calculates the properties of an array transducer, including its total length,
    element size, gap size, and element centroids.
    """

    @staticmethod
    def calculate(f, c, dl, gd, N):
        """
        Calculate array properties.

        Parameters:
            f (float): Frequency (in MHz).
            c (float): Wave speed (in m/s).
            dl (float): Element length divided by wavelength (d/λ).
            gd (float): Gap size divided by element length (g/d).
            N (int): Number of elements.

        Returns:
            tuple: (A, d, g, xc)
                - A (float): Total aperture size of the array (in mm).
                - d (float): Element size (in mm).
                - g (float): Gap size (in mm).
                - xc (numpy.ndarray): Centroids of array elements (in mm).
        """
        # Calculate element size (d)
        d = dl * c / (1000 * f)

        # Calculate gap size (g)
        g = gd * d

        # Calculate total aperture size (A)
        A = N * d + (N - 1) * g

        # Calculate centroids (xc)
        xc = np.array([(g + d) * ((2 * nn - 1) / 2 - N / 2) for nn in range(1, N + 1)])

        return A, d, g, xc

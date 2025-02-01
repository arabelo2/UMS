import numpy as np

class Elements:
    """
    Calculates the properties of an array transducer, including its total length,
    element size, gap size, and element centroids.
    """

    @staticmethod
    def calculate(f, c, dl, gd, N):
        """
        Compute the array properties, including aperture size, element size, gap, and centroids.

        Parameters:
            f (float): Frequency in MHz.
            c (float): Wave speed in the medium (m/s).
            dl (float): Element length divided by wavelength.
            gd (float): Gap between elements divided by element length.
            N (int): Number of elements.

        Returns:
            tuple: (A, d, g, xc) where:
                - A (float): Total aperture size.
                - d (float): Element length.
                - g (float): Gap size.
                - xc (np.ndarray): X-coordinates of the element centroids.
        """

        # Calculate element length and gap size
        λ = c / (f * 1e6)  # Convert MHz to Hz
        d = dl * λ
        g = gd * d  # Calculate gap size (g)

        # Compute aperture size
        A = N * d + (N - 1) * g

        # Compute element centroids
        xc = np.array([(m - (N - 1) / 2) * (d + g) for m in range(N)])

        return A, d, g, xc

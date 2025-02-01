import numpy as np
from domain.elements import Elements


class ElementsService:
    """
    Service for computing element properties in phased array models.
    """

    def calculate(self, f, c, dl, gd, N):
        """
        Compute the size of the array, element length, gap size, and element centroids.

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
        return Elements.calculate(f, c, dl, gd, N)

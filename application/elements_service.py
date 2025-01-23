from domain.elements import Elements


class ElementsService:
    """
    Service to calculate array transducer properties.
    """

    def calculate(self, f, c, dl, gd, N):
        """
        Compute array properties.

        Parameters:
            f (float): Frequency (in MHz).
            c (float): Wave speed (in m/s).
            dl (float): Element length divided by wavelength (d/λ).
            gd (float): Gap size divided by element length (g/d).
            N (int): Number of elements.

        Returns:
            tuple: (A, d, g, xc)
        """
        return Elements.calculate(f, c, dl, gd, N)

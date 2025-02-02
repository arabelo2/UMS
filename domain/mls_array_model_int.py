import numpy as np
from application.ls_2Dint_service import LS2DInterfaceService
from application.delay_laws2D_int_service import DelayLaws2DInterfaceService
from application.discrete_windows_service import DiscreteWindowsService

class MLSArrayModelInt:
    """Computes the normalized pressure wave field for a 1D phased array interacting with a fluid/fluid interface."""

    def __init__(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, Dt0, window_type):
        """
        Initialize parameters.

        Parameters:
            f (float): Frequency (MHz)
            d1 (float): Density of first medium (g/cm³)
            c1 (float): Wavespeed in first medium (m/s)
            d2 (float): Density of second medium (g/cm³)
            c2 (float): Wavespeed in second medium (m/s)
            M (int): Number of elements
            d (float): Element length (mm)
            g (float): Gap length (mm)
            angt (float): Array tilt angle (degrees)
            ang20 (float): Steering angle in second medium (degrees)
            DF (float): Focal depth (mm) (∞ for no focusing)
            Dt0 (float): Distance of array from interface (mm)
            window_type (str): Type of amplitude weighting function
        """
        self.f = f
        self.d1 = d1
        self.c1 = c1
        self.d2 = d2
        self.c2 = c2
        self.M = M
        self.d = d
        self.g = g
        self.angt = angt
        self.ang20 = ang20
        self.DF = DF
        self.Dt0 = Dt0
        self.window_type = window_type
        
        # Compute element half length
        self.b = self.d/2

        # Compute element positions
        self.s = self.d + self.g  # Pitch (element spacing)
        self.e = np.linspace(-((self.M - 1) / 2) * self.s, ((self.M - 1) / 2) * self.s, self.M)

        # Instantiate DelayLaws2DInterfaceService
        self.td_service = DelayLaws2DInterfaceService()
        self.td = self.td_service.compute_delays(self.M, self.s, self.angt, self.ang20, self.Dt0, self.DF, self.c1, self.c2)

        # Instantiate DiscreteWindowsService
        self.window_service = DiscreteWindowsService(self.M, self.window_type)
        self.Ct = self.window_service.calculate_weights()

        # Instantiate LS2DInterfaceService
        mat = (self.d1, self.c1, self.d2, self.c2)        
        self.ls_service = LS2DInterfaceService(self.b, self.f, list(mat), self.angt, self.Dt0)


    def compute_pressure_field(self, x, z):
        """
        Compute the total normalized pressure field.

        Parameters:
            x (np.ndarray): X-coordinates of the field (mm)
            z (np.ndarray): Z-coordinates of the field (mm)

        Returns:
            np.ndarray: Computed pressure field
        """
        xx, zz = np.meshgrid(x, z)  # Create 2D grid
        pressure = np.zeros_like(xx, dtype=np.complex128)

        for i in range(self.M):
            element_pressure = self.ls_service.calculate_pressure(xx, zz, self.e[i])
            pressure += self.Ct[i] * np.exp(1j * 2 * np.pi * self.f * self.td[i]) * element_pressure

        return np.abs(pressure)

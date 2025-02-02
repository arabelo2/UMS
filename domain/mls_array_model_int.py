import numpy as np
from domain.ls_2Dint import LS2DInterface
from domain.delay_laws2D_int import DelayLaws2DInterface
from domain.discrete_windows import DiscreteWindows

class MLSArrayModelInt:
    """Computes the normalized pressure wave field for a 1D phased array interacting with a fluid/fluid interface."""

    def __init__(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, window_type):
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
            DT0 (float): Distance of array from interface (mm)
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
        self.DT0 = DT0
        self.window_type = window_type

        # Compute element positions
        self.s = d + g  # Pitch (element spacing)
        self.e = np.linspace(-((M - 1) / 2) * self.s, ((M - 1) / 2) * self.s, M)

        # Compute time delays
        self.td_solver = DelayLaws2DInterface(M, self.s, self.angt, self.ang20, self.DT0, self.DF, self.c1, self.c2)
        self.td = self.td_solver.compute_delays()

        # Compute amplitude weights using window function
        self.window_solver = DiscreteWindows()
        self.Ct = self.window_solver.generate_weights(self.M, self.window_type)

        # Initialize LS2D solver
        self.ls_solver = LS2DInterface(self.f, self.d1, self.c1, self.d2, self.c2)

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
            element_pressure = self.ls_solver.compute_pressure(xx, zz, self.e[i])
            pressure += self.Ct[i] * np.exp(1j * 2 * np.pi * self.f * self.td[i]) * element_pressure

        return np.abs(pressure)

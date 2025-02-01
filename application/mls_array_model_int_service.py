import numpy as np
import matplotlib.pyplot as plt
from application.delay_laws2D_int_service import DelayLaws2DInterfaceService
from application.discrete_windows_service import DiscreteWindowsService
from application.ls_2Dint_service import LS2DInterfaceService
from application.elements_service import ElementsService


class MLSArrayModelInterfaceService:
    """
    Service to compute the normalized pressure wave field of an array passing
    through a fluid/fluid interface.
    """

    def compute_pressure(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, type_):
        """
        Compute the normalized pressure field for an ultrasonic phased array.

        Parameters:
            f (float): Frequency in MHz.
            d1 (float): Density of first medium (gm/cm^3).
            c1 (float): Wave speed in first medium (m/s).
            d2 (float): Density of second medium (gm/cm^3).
            c2 (float): Wave speed in second medium (m/s).
            M (int): Number of elements.
            d (float): Element length (mm).
            g (float): Gap length (mm).
            angt (float): Angle of the array.
            ang20 (float): Steering angle in second medium (degrees).
            DF (float): Focal depth (mm). Set to `np.inf` for no focusing.
            DT0 (float): Distance of the array from the interface (mm).
            type_ (str): Type of amplitude weighting function.

        Returns:
            np.ndarray: The computed pressure wave field.
        """

        # Half-length of element
        b = d / 2
        # Pitch of the array
        s = d + g
        # Material properties array
        mat = [d1, c1, d2, c2]

        # Initialize required services
        elements_service = ElementsService()
        delay_service = DelayLaws2DInterfaceService()
        window_service = DiscreteWindowsService()

        # ✅ Pass required arguments when initializing LS2DInterfaceService
        ls_2Dint_service = LS2DInterfaceService(b, f, mat, angt, DT0)

        # Compute element center distances
        e = elements_service.compute_element_positions(M, s)

        # Compute time delays
        td = delay_service.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option="n")

        # Compute delay exponential and amplitude weights
        delay = np.exp(1j * 2 * np.pi * f * td)
        Ct = window_service.compute_amplitude_weights(M, type_)

        # Generate 2D area in second medium for field calculations
        x = np.linspace(-5, 15, 200)
        z = np.linspace(1, 20, 200)
        xx, zz = np.meshgrid(x, z)

        # Compute normalized pressure field
        p = np.zeros_like(xx, dtype=np.complex128)
        for mm in range(M):
            p += Ct[mm] * delay[mm] * ls_2Dint_service.compute_pressure(xx, zz, e[mm])

        return x, z, np.abs(p)

    def plot_pressure_field(self, x, z, p):
        """
        Plot the computed pressure field as a 2D heatmap.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(
            p, extent=[x.min(), x.max(), z.min(), z.max()],
            origin="lower", aspect="auto", cmap="jet"
        )
        plt.colorbar(label="Normalized Pressure")
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("2D Pressure Field for MLS Array Model")
        plt.show()
        
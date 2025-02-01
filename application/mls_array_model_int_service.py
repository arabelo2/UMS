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

        # ✅ Extract centroids from ElementsService
        _, _, _, e = elements_service.calculate(f, c1, d / (c1 / (1000 * f)), g / d, M)

        # ✅ Pass required arguments when initializing LS2DInterfaceService
        ls_2Dint_service = LS2DInterfaceService(b, f, mat, angt, DT0)

        # Compute time delays
        td = delay_service.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option="n")

        # ✅ Use the correct method from DiscreteWindowsService
        Ct = window_service.calculate_weights(M, type_)

        # Compute delay exponential
        delay = np.exp(1j * 2 * np.pi * f * td)

        # Generate 2D area in second medium for field calculations
        x = np.linspace(-5, 15, 200)
        z = np.linspace(1, 20, 200)
        xx, zz = np.meshgrid(x, z)

        # Compute normalized pressure field
        p = np.zeros_like(xx, dtype=np.complex128)
        for mm in range(M):
            p += Ct[mm] * delay[mm] * ls_2Dint_service.calculate_pressure(xx, zz, e[mm])  # ✅ Corrected function call

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

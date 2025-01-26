# interface/mls_array_modeling.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.elements_service import ElementsService
from application.delay_laws2D_service import DelayLaws2DService
from application.discrete_windows_service import DiscreteWindowsService
from application.ls_2Dv_service import LS2DvService


def main():
    # Input parameters
    f = 5.0  # Frequency (MHz)
    c = 1480.0  # Wave speed (m/s)
    M = 32  # Number of elements
    dl = 0.5  # Element length divided by wavelength (d/λ)
    gd = 0.1  # Gap size divided by element length (g/d)
    Phi = 20.0  # Steering angle (degrees)
    F = np.inf  # Focal length (mm); F = inf for no focusing
    window_type = 'rect'  # Type of amplitude weighting function

    # Calculate array size, element size, gap size, and centroids
    elements_service = ElementsService()
    A, d, g, e = elements_service.calculate(f, c, dl, gd, M)
    b = d / 2  # Half-length of the source
    s = d + g  # Pitch (element size + gap)

    # Generate 2D area for field calculations
    z = np.linspace(1, 100 * dl, 500)  # z-coordinates (mm)
    x = np.linspace(-50 * dl, 50 * dl, 500)  # x-coordinates (mm)
    xx, zz = np.meshgrid(x, z)

    # Generate time delays and amplitude weights
    delay_service = DelayLaws2DService()
    td = delay_service.compute_delays(M, s, Phi, F, c)
    delay = np.exp(1j * 2 * np.pi * f * td)

    window_service = DiscreteWindowsService()
    Ct = window_service.get_amplitudes(M, window_type)

    # Generate normalized pressure wave field
    pressure = np.zeros_like(xx, dtype=np.complex128)
    ls_2Dv_service = LS2DvService()
    for mm in range(M):
        pressure += Ct[mm] * delay[mm] * ls_2Dv_service.calculate(b, f, c, e[mm], xx, zz)

    # Generate wave field image
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.abs(pressure),
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        cmap="jet"
    )
    plt.colorbar(label="Normalized Pressure")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("Pressure Wave Field")
    plt.show()


if __name__ == "__main__":
    main()

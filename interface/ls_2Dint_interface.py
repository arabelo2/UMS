# interface/ls_2Dint_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dint_service import LS2DInterfaceService

def main():
    # Parameters
    b = 3  # Half-length of the source (mm)
    f = 5  # Frequency (MHz)
    mat = [1, 1480, 7.9, 5900]  # Material properties [d1, c1, d2, c2]
    angt = 10.217  # Angle of the array (degrees)
    Dt0 = 50.8  # Distance of the array from the interface (mm)

    # Meshgrid for x and z
    x = np.linspace(0, 25, 200)  # x-coordinates (mm)
    z = np.linspace(1, 25, 200)  # z-coordinates (mm)
    xx, zz = np.meshgrid(x, z)

    # Initialize service
    service = LS2DInterfaceService(b, f, mat, angt, Dt0)

    # Compute normalized pressure over the grid
    p = np.zeros_like(xx, dtype=np.complex128)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p[i, j] = service.calculate_pressure(xx[i, j], zz[i, j], 0)

    # Plotting the result
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.abs(p),
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    plt.colorbar(label="Normalized Pressure |p|")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("2D Interface Pressure Distribution")
    plt.show()


if __name__ == "__main__":
    main()

# interface/ls_2Dint_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dint_service import LS2DInterfaceService
from application.ls_2Dv_service import LS2DvService


def main():
    # -------------------------------
    # On-Axis Pressure: Default Segment Size (1/10 Wavelength)
    # -------------------------------
    b = 3  # Half-length of the source (mm)
    f = 5  # Frequency (MHz)
    c = 1500  # Wave speed in the fluid (m/s)
    x = 0  # x-coordinate (mm)
    z = np.linspace(5, 80, 200)  # z-coordinates (mm)

    # Initialize the LS2DvService
    ls_2Dv_service = LS2DvService()

    # Compute normalized pressure with default segment size
    p_default = ls_2Dv_service.calculate(b, f, c, 0, x, z)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(p_default), label="|p| (Default Segment Size)")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("On-Axis Pressure (Default Segment Size)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------
    # On-Axis Pressure: Using 20 Segments (Segment Size = Wavelength)
    # -------------------------------
    # Compute normalized pressure with 20 segments
    p_20_segments = ls_2Dv_service.calculate(b, f, c, 0, x, z, Nopt=20)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(p_20_segments), label="|p| (20 Segments)")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("On-Axis Pressure (20 Segments)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------
    # 2D Interface Pressure Distribution
    # -------------------------------
    mat = [1, 1480, 7.9, 5900]  # Material properties [d1, c1, d2, c2]
    angt = 10.217  # Angle of the array (degrees)
    Dt0 = 50.8  # Distance of the array from the interface (mm)

    # Meshgrid for x and z
    x = np.linspace(0, 25, 200)  # x-coordinates (mm)
    z = np.linspace(1, 25, 200)  # z-coordinates (mm)
    xx, zz = np.meshgrid(x, z)

    # Initialize LS2DInterfaceService
    ls_2Dint_service = LS2DInterfaceService(b, f, mat, angt, Dt0)

    # Compute normalized pressure over the grid
    p_2d = np.zeros_like(xx, dtype=np.complex128)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p_2d[i, j] = ls_2Dint_service.calculate_pressure(xx[i, j], zz[i, j], 0)

    # Plot the 2D result
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.abs(p_2d),
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

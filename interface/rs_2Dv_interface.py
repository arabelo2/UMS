# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.rs_2Dv_service import RS2DvService

def main():
    # Initialize the RS2DvService
    rs_2Dv_service = RS2DvService()

    # -------------------------------
    # First Plot: 1D Normalized Pressure vs z
    # -------------------------------
    b = 6.35 / 2  # Half-length of the element (in mm)
    f = 5  # Frequency (in MHz)
    c = 1500  # Wave speed in the fluid (in m/s)
    e = 0  # Offset of the center of the element (in mm)
    x = 0  # x-coordinate (in mm)
    z = np.linspace(5, 200, 500)  # z-coordinates (in mm)

    # Compute normalized pressure for the 1D case
    p_1d = rs_2Dv_service.calculate(b, f, c, e, x, z)

    # Plot 1D result
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(p_1d), label="|p| (Normalized Pressure)")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("Normalized Pressure vs z (Rayleigh-Sommerfeld)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------
    # Second Plot: 2D Normalized Pressure
    # -------------------------------
    x = np.linspace(-10, 10, 200)  # x-coordinates (in mm)
    z = np.linspace(1, 20, 200)  # z-coordinates (in mm)
    xx, zz = np.meshgrid(x, z)

    b = 1  # Half-length of the element (in mm)

    # Compute normalized pressure for the 2D case
    p_2d = rs_2Dv_service.calculate(b, f, c, e, xx, zz)

    # Plot 2D result
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.abs(p_2d),
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        cmap="jet"
    )
    plt.colorbar(label="Normalized Pressure")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("2D Normalized Pressure Field (Rayleigh-Sommerfeld)")
    plt.show()


if __name__ == "__main__":
    main()

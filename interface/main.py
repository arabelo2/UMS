import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.rs_2Dv_service import RS2DvService
from application.fresnel_2D_service import Fresnel2DService


def main():
    # Parameters
    b = 6  # Half-length of the element (in mm)
    f = 5  # Frequency (in MHz)
    c = 1500  # Wave speed in the fluid (in m/sec)

    # Input grid
    z = 60  # Fixed z-coordinate (in mm)
    x = np.linspace(-10, 10, 200)  # x-coordinates (in mm)

    # Initialize service
    service = Fresnel2DService(b, f, c)

    # Compute normalized pressure
    pressure = service.compute_pressure(x, z)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(x, np.abs(pressure), label="|p|", color="blue")
    plt.xlabel("x (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title(f"Normalized Pressure Field at z = {z} mm")  # Dynamically update title
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

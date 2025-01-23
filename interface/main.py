# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.np_gauss_2D_service import NPGauss2DService


def main():
    # Parameters for the test
    b = 3  # Half-length of the element (in mm)
    f = 5  # Frequency (in MHz)
    c = 1500  # Wave speed in the fluid (in m/s)
    e = 0  # Offset in the x-direction (in mm)
    x = 0  # x-coordinate (in mm)
    z = np.linspace(5, 80, 200)  # z-coordinates (in mm)

    # Initialize the service
    np_gauss_service = NPGauss2DService()

    # Compute the normalized pressure field
    p = np_gauss_service.calculate(b, f, c, e, x, z)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(p), label="|p| (Normalized Pressure)")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("Normalized Pressure vs z")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

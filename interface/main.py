# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dv_service import LS2DvService


def main():
    # Parameters for the test case
    b = 3  # Half-length of the source (in mm)
    f = 5  # Frequency (in MHz)
    c = 1500  # Wave speed (in m/s)
    x = 0  # x-coordinate (in mm)
    z = np.linspace(5, 80, 200)  # z-coordinates (in mm)

    # Initialize the LS2DvService
    ls_2Dv_service = LS2DvService()

    # Compute the normalized pressure
    p = ls_2Dv_service.calculate(b, f, c, 0, x, z)

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

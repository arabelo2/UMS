import numpy as np
import matplotlib.pyplot as plt
from application.rs_2Dv_service import RS2DvService


def main():
    # Parameters
    b = 3  # Half-length of the element (in mm)
    f = 5  # Frequency (in MHz)
    c = 1500  # Wave speed in the fluid (in m/s)
    e = 0  # Offset of the center of the element (in mm)
    x = 0  # x-coordinate (in mm)
    z = np.linspace(5, 80, 200)  # z-coordinates (in mm)

    # Initialize the RS2DvService
    rs_2Dv_service = RS2DvService()

    # Compute normalized pressure
    p = rs_2Dv_service.calculate(b, f, c, e, x, z)

    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(p), label="|p| (Normalized Pressure)")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("Normalized Pressure vs z (Rayleigh-Sommerfeld)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

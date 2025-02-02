import numpy as np
import matplotlib.pyplot as plt
from application.mls_array_model_int_service import MLSArrayModelIntService

def main():
    """Main function to compute and visualize phased array interactions with an interface."""

    # ------------ Define input parameters ------------------
    f = 5  # Frequency (MHz)
    d1 = 1.0  # Density of first medium (g/cm³)
    c1 = 1480  # Wavespeed (m/s) in first medium
    d2 = 7.9  # Density of second medium (g/cm³)
    c2 = 5900  # Wavespeed (m/s) in second medium
    M = 32  # Number of elements
    d = 0.25  # Element length (mm)
    g = 0.05  # Gap length (mm)
    angt = 0  # Array tilt angle (degrees)
    ang20 = 30.0  # Steering angle in second medium (degrees)
    DF = 8  # Focal depth (mm) (∞ for no focusing)
    DT0 = 25.4  # Distance of array from interface (mm)
    window_type = "rect"  # Type of amplitude weighting function

    # Define field grid
    x = np.linspace(-5, 15, 200)
    z = np.linspace(1, 20, 200)

    # Compute Pressure Field
    service = MLSArrayModelIntService(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, window_type)
    pressure = service.compute_pressure(x, z)

    # Plot Results
    plt.figure(figsize=(8, 6))
    plt.imshow(pressure, extent=[x.min(), x.max(), z.min(), z.max()], origin="lower", aspect="auto", cmap="jet")
    plt.colorbar(label="Normalized Pressure |p|")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("1D Phased Array Interaction with Fluid Interface")
    plt.show()

if __name__ == "__main__":
    main()
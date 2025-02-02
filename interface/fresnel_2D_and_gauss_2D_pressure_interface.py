import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.fresnel_2D_service import Fresnel2DService
from application.gauss_2D_service import Gauss2DService

def main():
    """Main function to test Fresnel_2D and Gauss_2D pressure calculations."""

    # ------------ Define input parameters ------------------
    b = 6  # Half-length of the transducer (mm)
    f = 5  # Frequency (MHz)
    c = 1500  # Speed of sound (m/s)
    z = 60  # Fixed depth (mm)

    # Compute Fresnel_2D Pressure
    x_fresnel = np.linspace(-10, 10, 200)  # X range for Fresnel
    fresnel_service = Fresnel2DService(b, f, c)
    p_fresnel = fresnel_service.compute_pressure(x_fresnel, z)

    # Compute Gauss_2D Pressure
    x_gauss = np.linspace(-10, 10, 40)  # X range for Gauss (fewer points)
    gauss_service = Gauss2DService(b, f, c)
    p_gauss = gauss_service.compute_pressure(x_gauss, z)

    # Plot Fresnel and Gauss Pressures
    plt.figure(figsize=(8, 6))
    
    # Fresnel 2D plot
    plt.plot(x_fresnel, np.abs(p_fresnel), label="Fresnel 2D", color="blue")

    # Gauss 2D plot (discrete points)
    plt.plot(x_gauss, np.abs(p_gauss), 'o', label="Gauss 2D", color="red")

    # Formatting
    plt.xlabel("x (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("Comparison: Fresnel 2D versus Gauss 2D Pressure")
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

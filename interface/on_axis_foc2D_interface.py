import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.on_axis_foc2D_service import OnAxisFocusedPistonService

def main():
    """Main function to test on-axis focused piston modeling."""

    # Define parameters
    b = 6  # Half-length of the transducer (mm)
    R = 100  # Focal length (mm)
    f = 5  # Frequency (MHz)
    c = 1480  # Speed of sound (m/s)

    # Define axial depth range
    z = np.linspace(20, 400, 500)  # Axial positions (mm)

    # Initialize service
    service = OnAxisFocusedPistonService(b, R, f, c)

    # Compute pressure
    pressure = service.compute_pressure(z)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(z, np.abs(pressure), label="|p| (Normalized Pressure)", color="blue")
    plt.xlabel("z (mm)")
    plt.ylabel("|p| (Normalized Pressure)")
    plt.title("On-Axis Normalized Pressure for Focused Piston")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

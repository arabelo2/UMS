import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.mls_array_model_int_service import MLSArrayModelInterfaceService


def main():
    # Parameters
    f = 5  # Frequency (MHz)
    d1 = 1.0  # Density of first medium (gm/cm^3)
    c1 = 1480  # Wave speed in first medium (m/s)
    d2 = 7.9  # Density of second medium (gm/cm^3)
    c2 = 5900  # Wave speed in second medium (m/s)
    M = 32  # Number of elements
    d = 0.25  # Element length (mm)
    g = 0.05  # Gap length (mm)
    angt = 0  # Angle of array
    ang20 = 30.0  # Steering angle (degrees in second medium)
    DF = 8  # Focal depth (mm); `np.inf` for no focusing
    DT0 = 25.4  # Distance from interface (mm)
    type_ = "rect"  # Type of amplitude weighting function

    # ✅ Generate 2D area for field calculations (Moved here from service)
    x = np.linspace(-5, 15, 200)
    z = np.linspace(1, 20, 200)
    xx, zz = np.meshgrid(x, z)

    # Initialize service
    service = MLSArrayModelInterfaceService()

    # Compute the pressure field
    p = service.compute_pressure(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, type_, xx, zz)

    # Plot the result
    service.plot_pressure_field(x, z, p)


if __name__ == "__main__":
    main()

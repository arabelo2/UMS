#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

# Import the elements calculator from the domain module.
from domain.elements import ElementsCalculator
# Import the modeling service from the application module.
from application.mls_array_modeling_service import run_mls_array_modeling_service

def main():
    # Input parameters (same as your MATLAB script)
    f     = 5         # frequency (MHz)
    c     = 1480      # wave speed (m/s)
    M     = 32        # number of elements
    dl    = 0.5       # element length (d) divided by wavelength
    gd    = 0.1       # gap (g) divided by element length
    Phi   = 20.0      # steering angle (degrees)
    F     = np.inf    # focal length (mm), np.inf for steering-only
    wtype = 'rect'    # type of amplitude weighting function

    # Generate 2-D mesh grid for field calculations (in mm)
    z = np.linspace(1, 100 * dl, 500)
    x = np.linspace(-50 * dl, 50 * dl, 500)
    xx, zz = np.meshgrid(x, z)

    # Use the modeling service to compute the pressure field.
    # run_mls_array_modeling_service() handles:
    #   - Computing the array parameters using ElementsCalculator
    #   - Computing the delay laws, amplitude weights, and individual element fields.
    p, A, d, g, e = run_mls_array_modeling_service(f, c, M, dl, gd, Phi, F, wtype, xx, zz)

    # Plot the normalized pressure magnitude field.
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(p), extent=[x[0], x[-1], z[0], z[-1]], origin='lower', aspect='auto')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.title('MLS Array Modeling Pressure Field')
    plt.colorbar(label='Normalized Pressure Magnitude')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.delay_laws2D_service import DelayLaws2DService

def main():
    # Parameters
    M = 16  # Number of elements
    s = 0.5  # Pitch (in mm)
    Phi = 15  # Steering angle (in degrees)
    # F = np.inf  # Focal length (set to infinity for steering only) 
    F = 20  # Focal Length = {F} mm
    c = 1480  # Wave speed (in m/s)

    # Initialize service
    service = DelayLaws2DService()

    # Compute time delays
    td = service.compute_delays(M, s, Phi, F, c)

    # Display results
    print("Time Delays (microseconds):")
    for i, delay in enumerate(td, start=1):
        print(f"Element {i}: {delay:.4f} µs")

    # Plot the delays
    plt.figure(figsize=(8, 6))
    plt.stem(range(1, M + 1), td)  # Generate stem plot
    plt.xlabel("Element Index")
    plt.ylabel("Time Delay (µs)")

    # Set plot title based on focal length
    if np.isinf(F):
        plt.title("Time Delays for Array Elements (Steering Only)")
    else:
        plt.title(f"Time Delays for Array Elements (Focal Length = {F} mm)")

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

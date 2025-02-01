# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.delay_laws2D_int_service import DelayLaws2DInterfaceService


def main():
    # Parameters
    M = 16  # Number of elements
    s = 0.5  # Pitch (mm)
    angt = 10  # Array angle with interface (degrees)
    ang20 = 15  # Refracted angle in second medium (degrees)
    DT0 = 50.8  # Height of array center above interface (mm)
    DF = 100  # Depth in the second medium (mm); Use np.inf for steering only
    c1 = 1480  # Wave speed in first medium (m/s)
    c2 = 5900  # Wave speed in second medium (m/s)

    # Ensure plt_option is always defined
    plt_option = globals().get("plt_option", "n")  # Default to 'n' if not set

    # Initialize the service
    service = DelayLaws2DInterfaceService()

    # Compute the delay laws
    td = service.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)

    # Display results
    print("Time Delays (microseconds):")
    for i, delay in enumerate(td, start=1):
        print(f"Element {i}: {delay:.4f} µs")

    # Plot the delays
    plt.figure(figsize=(8, 6))
    plt.stem(range(1, M + 1), td)
    plt.xlabel("Element Index")
    plt.ylabel("Time Delay (µs)")
    plt.title(f"Time Delays for 2D Interface (DF = {DF} mm)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

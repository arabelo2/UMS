# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from application.discrete_windows_service import DiscreteWindowsService


def main():
    # Parameters
    M = 16  # Number of elements
    window_type = 'tri'  # Window type

    # Initialize service
    service = DiscreteWindowsService()

    # Compute apodization amplitudes
    amplitudes = service.get_amplitudes(M, window_type)

    # Display results
    print(f"Apodization Amplitudes for {window_type} window:")
    for i, amp in enumerate(amplitudes, start=1):
        print(f"Element {i}: {amp:.4f}")

    # Plot the amplitudes
    plt.figure(figsize=(8, 6))
    plt.stem(range(1, M + 1), amplitudes)  # Removed `use_line_collection`
    plt.xlabel("Element Index")
    plt.ylabel("Amplitude")
    plt.title(f"Apodization Amplitudes for {window_type} Window")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

# interface/discrete_windows_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from application.discrete_windows_service import DiscreteWindowsService

def main():
    """Main function to test discrete windowing functions."""

    # Define parameters
    M = 16  # Number of elements
    window_type = "Blk"  # Choose: 'cos', 'Han', 'Ham', 'Blk', 'tri', 'rect'

    # Compute Window Weights
    service = DiscreteWindowsService(M, window_type)
    weights = service.calculate_weights()

    # Plot Window Weights
    plt.figure(figsize=(8, 6))
    plt.stem(range(1, M + 1), weights, basefmt=" ", use_line_collection=True)

    # Formatting
    plt.xlabel("Element Index")
    plt.ylabel("Amplitude Weight")
    plt.title(f"Discrete Windowing Function: {window_type}")
    plt.grid(True)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

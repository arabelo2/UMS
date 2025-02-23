#!/usr/bin/env python
# interface/mls_array_modeling_gauss_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import matplotlib.pyplot as plt
import numpy as np
from application.mls_array_modeling_gauss_service import run_mls_array_modeling_gauss
from interface.cli_utils import safe_float, parse_array  # Use only the available utility functions

def main():
    parser = argparse.ArgumentParser(
        description="MLS Array Modeling with Gaussian beam approximation.",
        epilog=(
            "Example usage:\n"
            "  python interface/mls_array_modeling_gauss_interface.py "
            "--frequency 5 --wavespeed 1480 --elements 32 --dl 0.5 --gd 0.1 "
            "--phi 20 --focal inf --window rect\n"
            "Defaults simulate the Gaussian MLS Array Modeling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--frequency", type=safe_float, default=5.0,
                        help="Frequency in MHz. Default: 5.0")
    parser.add_argument("--wavespeed", type=safe_float, default=1480.0,
                        help="Wave speed in m/sec. Default: 1480")
    parser.add_argument("--elements", type=int, default=32,
                        help="Number of elements. Default: 32")
    parser.add_argument("--dl", type=safe_float, default=0.5,
                        help="Normalized element length (d / wavelength). Default: 0.5")
    parser.add_argument("--gd", type=safe_float, default=0.1,
                        help="Normalized gap between elements (g / d). Default: 0.1")
    parser.add_argument("--phi", type=safe_float, default=20.0,
                        help="Steering angle in degrees. Default: 20")
    parser.add_argument("--focal", type=safe_float, default=float('inf'),
                        help="Focal length in mm (use 'inf' for no focusing). Default: inf")
    parser.add_argument("--window", type=str, default="rect",
                        choices=["cos", "Han", "Ham", "Blk", "tri", "rect"],
                        help="Amplitude weighting function type. Default: rect")
    parser.add_argument("--plot", type=str, choices=["Y", "N"], default="Y",
                        help="Show plot? Y/N. Default: Y")
    
    args = parser.parse_args()

    # Call the Gaussian MLS array modeling service
    result = run_mls_array_modeling_gauss(
        args.frequency, args.wavespeed, args.elements,
        args.dl, args.gd, args.phi, args.focal, args.window
    )
    
    p = result['p']
    x = result['x']
    z = result['z']
    
    # Save the results to a file
    outfile = "mls_array_modeling_gauss_output.txt"
    with open(outfile, "w") as f:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                f.write(f"{p[i, j].real:.6f}+{p[i, j].imag:.6f}j\t")
            f.write("\n")
    print(f"Results saved to {outfile}")

    # Plot the pressure field if requested
    if args.plot.upper() == "Y":
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(p), cmap="jet",
                   extent=[x.min(), x.max(), z.max(), z.min()],
                   aspect='equal')
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("Normalized Pressure Field (Gaussian MLS Array Modeling)")
        plt.colorbar(label="Pressure Magnitude")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

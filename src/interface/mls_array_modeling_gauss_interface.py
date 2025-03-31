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
            "--f 5 --c 1480 --M 32 --dl 0.5 --gd 0.1 --Phi 20 --F inf --wtype rect --plot Y\n"
            "Defaults simulate the Gaussian MLS Array Modeling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency in MHz. Default: 5.0")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed in m/sec. Default: 1480")
    parser.add_argument("--M", type=int, default=32,
                        help="Number of elements. Default: 32")
    parser.add_argument("--dl", type=safe_float, default=0.5,
                        help="Normalized element length (d / wavelength). Default: 0.5")
    parser.add_argument("--gd", type=safe_float, default=0.1,
                        help="Normalized gap between elements (g / d). Default: 0.1")
    parser.add_argument("--Phi", type=safe_float, default=20.0,
                        help="Steering angle in degrees. Default: 20")
    parser.add_argument("--F", type=safe_float, default=float('inf'),
                        help="Focal length in mm (use 'inf' for no focusing). Default: inf")
    parser.add_argument("--wtype", type=str, default="rect",
                        choices=["cos", "Han", "Ham", "Blk", "tri", "rect"],
                        help="Amplitude weighting function type. Default: rect")
    parser.add_argument("--plot", type=str, choices=["Y", "N"], default="Y",
                        help="Show plot? Y/N. Default: Y")
    
    args = parser.parse_args()

    # Call the Gaussian MLS array modeling service with the updated parameters
    result = run_mls_array_modeling_gauss(
        args.f, args.c, args.M, args.dl, args.gd, args.Phi, args.F, args.wtype
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
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(p), cmap="jet",
                   extent=[x.min(), x.max(), z.max(), z.min()],
                   aspect='auto')
        plt.xlabel("x (mm)", fontsize=16)
        plt.ylabel("z (mm)", fontsize=16)

        # Dynamic title based on parameters
        if np.isinf(args.F):
            title = f"Gaussian MLS Steered Beam (Φ={args.Phi}°, F=∞, M={args.M}, f={args.f} MHz, Window={args.wtype})"
        else:
            title = f"Gaussian MLS Steered + Focused Beam (Φ={args.Phi}°, F={args.F} mm, M={args.M}, f={args.f} MHz, Window={args.wtype})"
        plt.title(title, fontsize=18, linespacing=1.2)

        cbar = plt.colorbar()
        cbar.set_label("Normalized pressure magnitude", fontsize=16, linespacing=1.2)
        cbar.ax.tick_params(labelsize=14)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

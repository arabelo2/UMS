#!/usr/bin/env python3
# interface/mls_array_modeling_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from domain.elements import ElementsCalculator
from application.mls_array_modeling_service import run_mls_array_modeling_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Simulate the MLS Array Modeling process.",
        epilog=(
            "Example usage:\n"
            "  python interface/mls_array_modeling_interface.py --f 5 --c 1480 --M 32 --dl 0.5 --gd 0.1 --Phi 20 --F inf --wtype rect --plot Y\n"
            "Default parameters provided simulate the MLS Array Modeling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--f", type=safe_float, default=5, help="Frequency in MHz. Default: 5")
    parser.add_argument("--c", type=safe_float, default=1480, help="Wave speed in m/s. Default: 1480")
    parser.add_argument("--M", type=int, default=32, help="Number of elements. Default: 32")
    parser.add_argument("--dl", type=safe_float, default=0.5, help="Element length divided by wavelength. Default: 0.5")
    parser.add_argument("--gd", type=safe_float, default=0.1, help="Gap size divided by element length. Default: 0.1")
    parser.add_argument("--Phi", type=safe_float, default=20, help="Steering angle in degrees. Default: 20")
    parser.add_argument("--F", type=safe_float, default=float('inf'), help="Focal length in mm. Default: inf")
    parser.add_argument("--wtype", type=str, default="rect", choices=["cos", "Han", "Ham", "Blk", "tri", "rect"],
                        help="Window type. Default: rect")
    parser.add_argument("--plot", type=str, choices=["Y", "N"], default="Y", help="Show plot? Y/N. Default: Y")

    args = parser.parse_args()

    # Generate 2D mesh grid
    z = np.linspace(1, 100 * args.dl, 500)
    x = np.linspace(-50 * args.dl, 50 * args.dl, 500)
    xx, zz = np.meshgrid(x, z)

    # Calculate elements directly using the domain layer
    elements_calc = ElementsCalculator(args.f, args.c, args.dl, args.gd, args.M)
    A, d, g, e = elements_calc.calculate()

    # Run the modeling process using the application layer
    p, A, d, g, e = run_mls_array_modeling_service(
        args.f, args.c, args.M, args.dl, args.gd, args.Phi, args.F, args.wtype, xx, zz
    )

    # ------------------------------------------------------------------
    # Save the results to a file (matrix-style: one row per grid-row)
    # ------------------------------------------------------------------
    outfile = args.outfile            # honour --outfile from the CLI
    try:
        with open(outfile, "w") as f:
            for row in p:             # p is a 2-D array
                formatted_row = "\t".join(
                    f"{val.real:+.6e}{val.imag:+.6e}j" for val in row
                )
                f.write(formatted_row + "\n")
        print(f"[OK] Complex pressure matrix saved in '{outfile}'")
    except Exception as exc:
        print(f"[ERROR] Could not write '{outfile}': {exc}")

    # Plot the results if requested
    if args.plot.upper() == "Y":
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(p), cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
        plt.xlabel("x (mm)", fontsize=16)
        plt.ylabel("z (mm)", fontsize=16)

        # Dynamically build title
        if np.isinf(args.F):
            title = f"MLS Steered Beam (Φ={args.Phi}°, F=∞, M={args.M}, f={args.f} MHz, Window={args.wtype})"
        else:
            title = f"MLS Steered + Focused Beam (Φ={args.Phi}°, F={args.F} mm, M={args.M}, f={args.f} MHz, Window={args.wtype})"
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

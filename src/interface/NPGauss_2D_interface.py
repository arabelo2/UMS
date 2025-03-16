#!/usr/bin/env python3
# interface/NPGauss_2D_interface.py

"""
Module: NPGauss_2D_interface.py
Layer: Interface

Provides a CLI for computing the normalized pressure field of a 1-D element
using a non-paraxial multi-Gaussian beam model (NPGauss_2D).

Default values:
  b = 6 (mm)
  f = 5 (MHz)
  c = 1500 (m/s)
  e = 0 (mm) offset
  x = "-10,10,200" -> np.linspace(-10,10,200)
  z = 60 (mm)
  plot = Y (show plot by default)

Example usage:
  1) python interface/NPGauss_2D_interface.py --b 6 --f 5 --c 1500 --e 2.0 --x="-10,10,200" --z 60 --plot Y
  2) python interface/NPGauss_2D_interface.py                (all defaults)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.NPGauss_2D_service import run_np_gauss_2D_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure using the Non-Paraxial Gauss 2D model.",
        epilog=(
            "Example usage:\n"
            "  1) python interface/NPGauss_2D_interface.py --b 6 --f 5 --c 1500 --e 2.0 --x=\"-10,10,200\" --z 60 --plot Y\n"
            "  2) python interface/NPGauss_2D_interface.py                (defaults)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--b", type=safe_float, default=6.0,
                        help="Half-length of the element (mm). Default=6")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default=5")
    parser.add_argument("--c", type=safe_float, default=1500.0,
                        help="Wave speed (m/s). Default=1500")
    parser.add_argument("--e", type=safe_float, default=0.0,
                        help="Offset of the element's center in x-direction (mm). Default=0")
    parser.add_argument("--x", type=str, default="-10,10,200",
                        help="x-coordinates as 'start,stop,num_points' or comma list. Default='-10,10,200'.")
    parser.add_argument("--z", type=safe_float, default=60.0,
                        help="z-coordinate (mm). Default=60")
    parser.add_argument("--plot", type=str, choices=["Y","N","y","n"], default="Y",
                        help="If 'Y', plot abs(p) vs x. Default='Y'")
    parser.add_argument("--outfile", type=str, default="np_gauss_2D_output.txt",
                        help="Output file name. Default=np_gauss_2D_output.txt")

    args = parser.parse_args()

    # Parse x array, keep z as a single float for simplicity
    x_vals = parse_array(args.x)

    try:
        p = run_np_gauss_2D_service(args.b, args.f, args.c, args.e, x_vals, args.z)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the complex results to outfile
    with open(args.outfile, "w") as f:
        for val in p:
            f.write(f"{val.real:.6f}+{val.imag:.6f}j\n")
    print(f"Results saved to {args.outfile}")

    # Optionally plot
    if args.plot.upper() == "Y":
        plt.figure(figsize=(8,5))
        plt.plot(x_vals, np.abs(p), 'b-', label="NPGauss 2D")
        plt.xlabel("x (mm)")
        plt.ylabel("Normalized Pressure Magnitude")
        plt.title("Non-Paraxial Gauss 2D Pressure Field")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# interface/delay_laws2D_interface.py

"""
Module: delay_laws2D_interface.py
Layer: Interface

Provides a CLI to compute time delays for a 1D array using the delay_laws2D model.

Default values:
  M   = 16
  s   = 0.5 mm
  Phi = 30 degrees
  F   = inf  (steering only)
  c   = 1480 m/s
  plot = Y

Example usage:
  1) Steering only:
     python interface/delay_laws2D_interface.py --M 16 --s 0.5 --Phi 30 --F inf --c 1480 --plot Y

  2) Steering + focusing:
     python interface/delay_laws2D_interface.py --M 16 --s 0.5 --Phi 0 --F 15 --c 1480 --plot Y
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.delay_laws2D_service import run_delay_laws2D_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute time delays (us) for a 1-D array using delay_laws2D.",
        epilog=(
            "Example usage:\n"
            "  1) Steering only:\n"
            "     python interface/delay_laws2D_interface.py --M 16 --s 0.5 --Phi 30 --F inf --c 1480 --plot Y\n\n"
            "  2) Steering + focusing:\n"
            "     python interface/delay_laws2D_interface.py --M 16 --s 0.5 --Phi 0 --F 15 --c 1480 --plot Y\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--M", type=int, default=16,
                        help="Number of elements. Default=16.")
    parser.add_argument("--s", type=safe_float, default=0.5,
                        help="Pitch in mm. Default=0.5.")
    parser.add_argument("--Phi", type=safe_float, default=30.0,
                        help="Steering angle in degrees. Default=30.0.")
    parser.add_argument("--F", type=safe_float, default=float('inf'),
                        help="Focal distance in mm. Use inf for steering only. Default=inf.")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default=1480.")
    parser.add_argument("--outfile", type=str, default="delay_laws2D_output.txt",
                        help="Output file to save the time delays. Default=delay_laws2D_output.txt")
    parser.add_argument("--plot", type=str, choices=["Y","N","y","n"], default="Y",
                        help="Display a stem plot: 'Y'/'N' (case-insensitive). Default='Y'")

    args = parser.parse_args()

    try:
        td = run_delay_laws2D_service(args.M, args.s, args.Phi, args.F, args.c)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save results to file
    with open(args.outfile, "w") as f:
        for i, val in enumerate(td):
            f.write(f"Element {i+1}: {val:.6f} us\n")
    print(f"Time delays saved to {args.outfile}")

    # If plot is requested
    if args.plot.upper() == "Y":
        # Determine the plot title
        if math.isinf(args.F):
            plot_title = "Delay Laws 2D - Steering Only"
        else:
            plot_title = "Delay Laws 2D - Steering + Focusing"

        plt.figure(figsize=(8, 5))
        x_coords = np.arange(1, len(td) + 1)
        plt.stem(x_coords, td, linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.xlabel("Element index")
        plt.ylabel("Time Delay (microseconds)")
        plt.title(plot_title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

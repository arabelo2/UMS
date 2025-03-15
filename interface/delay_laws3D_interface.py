#!/usr/bin/env python3
"""
Module: delay_laws3D_interface.py
Layer: Interface

Provides a CLI to compute time delays for a 2D array using delay_laws3D.

Default values:
  M     = 8
  N     = 8
  sx    = 0.15 mm
  sy    = 0.15 mm
  theta = 20 degrees
  phi   = 0 degrees
  F     = inf  (steering only)
  c     = 1480 m/s
  plot  = Y

Example usage:
  1) Steering only:
     python interface/delay_laws3D_interface.py --M 8 --N 8 --sx 0.15 --sy 0.15 --theta 20 --phi 0 --F inf --c 1480 --plot Y

  2) Steering + focusing:
     python interface/delay_laws3D_interface.py --M 8 --N 8 --sx 0.15 --sy 0.15 --theta 20 --phi 0 --F 11 --c 1480 --plot Y
"""

import sys
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from application.delay_laws3D_service import run_delay_laws3D_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute time delays (us) for a 2D array using delay_laws3D.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--M", type=int, default=8,
                        help="Number of elements in x-direction. Default=8.")
    parser.add_argument("--N", type=int, default=8,
                        help="Number of elements in y-direction. Default=8.")
    parser.add_argument("--sx", type=safe_float, default=0.15,
                        help="Pitch in x-direction (mm). Default=0.15.")
    parser.add_argument("--sy", type=safe_float, default=0.15,
                        help="Pitch in y-direction (mm). Default=0.15.")
    parser.add_argument("--theta", type=safe_float, default=20.0,
                        help="Steering angle theta (degrees). Default=20.")
    parser.add_argument("--phi", type=safe_float, default=0.0,
                        help="Steering angle phi (degrees). Default=0.")
    parser.add_argument("--F", type=safe_float, default=float('inf'),
                        help="Focal distance in mm (use inf for steering only). Default=inf.")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default=1480.")
    parser.add_argument("--outfile", type=str, default="delay_laws3D_output.txt",
                        help="Output file to save the time delays. Default=delay_laws3D_output.txt")
    parser.add_argument("--plot", type=str, choices=["Y","N","y","n"], default="Y",
                        help="Display a heatmap plot: 'Y'/'N'. Default=Y.")
    
    args = parser.parse_args()
    
    try:
        td = run_delay_laws3D_service(args.M, args.N, args.sx, args.sy,
                                      args.theta, args.phi, args.F, args.c)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save results to file
    with open(args.outfile, "w") as f:
        for i in range(args.M):
            for j in range(args.N):
                f.write(f"Element ({i+1},{j+1}): {td[i, j]:.6f} us\n")
    print(f"Time delays saved to {args.outfile}")
    
    # Plot the delay matrix as a heatmap if requested.
    if args.plot.upper() == "Y":
        if math.isinf(args.F):
            plot_title = "Delay Laws 3D - Steering Only"
        else:
            plot_title = "Delay Laws 3D - Steering + Focusing"
        plt.figure(figsize=(8, 6))
        plt.imshow(td, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label="Time Delay (us)")
        plt.xlabel("Element index (y-direction)")
        plt.ylabel("Element index (x-direction)")
        plt.title(plot_title)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

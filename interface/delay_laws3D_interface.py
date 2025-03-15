#!/usr/bin/env python3
"""
Module: delay_laws3D_interface.py
Layer: Interface

Provides a CLI to compute time delays for a 2D array using delay_laws3D
and plot them in a 3D stem plot (similar to MATLAB's stem3).

Default values:
  M     = 8
  N     = 16
  sx    = 0.5  mm
  sy    = 0.5  mm
  theta = 0    degrees
  phi   = 0    degrees
  F     = 10   mm  (use inf for steering only)
  c     = 1480 m/s
  plot  = Y
  elev  = 16   (camera elevation)
  azim  = -82  (camera azimuth)

Example usage:
  python interface/delay_laws3D_interface.py \
    --M 8 --N 16 --sx 0.5 --sy 0.5 --theta 0 --phi 0 --F 10 --c 1480 \
    --plot Y --elev 16 --azim -82
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.delay_laws3D_service import run_delay_laws3D_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute time delays (us) for a 2D array using delay_laws3D, then display a 3D stem plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Define arguments with updated defaults and new camera parameters
    parser.add_argument("--M", type=int, default=8,
                        help="Number of elements in x-direction. Default=8.")
    parser.add_argument("--N", type=int, default=16,
                        help="Number of elements in y-direction. Default=16.")
    parser.add_argument("--sx", type=safe_float, default=0.5,
                        help="Pitch in x-direction (mm). Default=0.5.")
    parser.add_argument("--sy", type=safe_float, default=0.5,
                        help="Pitch in y-direction (mm). Default=0.5.")
    parser.add_argument("--theta", type=safe_float, default=0.0,
                        help="Steering angle theta (degrees). Default=0.")
    parser.add_argument("--phi", type=safe_float, default=0.0,
                        help="Steering angle phi (degrees). Default=0.")
    parser.add_argument("--F", type=safe_float, default=10.0,
                        help="Focal distance in mm (use inf for steering only). Default=10.")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default=1480.")
    parser.add_argument("--outfile", type=str, default="delay_laws3D_output.txt",
                        help="Output file to save the time delays. Default=delay_laws3D_output.txt")
    parser.add_argument("--plot", type=str, choices=["Y","N","y","n"], default="Y",
                        help="Display a 3D stem plot: 'Y'/'N'. Default=Y.")
    parser.add_argument("--elev", type=safe_float, default=16.0,
                        help="Camera elevation for 3D plot. Default=16.")
    parser.add_argument("--azim", type=safe_float, default=-82.0,
                        help="Camera azimuth for 3D plot. Default=-82.")
    
    args = parser.parse_args()
    
    # Run the 3D delay calculation
    try:
        td = run_delay_laws3D_service(
            args.M, args.N, args.sx, args.sy,
            args.theta, args.phi, args.F, args.c
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save results to file
    with open(args.outfile, "w") as f:
        for i in range(args.M):
            for j in range(args.N):
                f.write(f"Element ({i+1},{j+1}): {td[i, j]:.6f} us\n")
    print(f"Time delays saved to {args.outfile}")
    
    # Plot a 3D stem if requested
    if args.plot.upper() == "Y":
        plot_title = "Delay Laws 3D - 3D Stem Plot"
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Generate grid indices for plotting
        X, Y = np.meshgrid(range(args.N), range(args.M))

        # Emulate MATLAB's stem3 by drawing vertical lines from z=0 to z=td[i,j].
        for i in range(args.M):
            for j in range(args.N):
                ax.plot(
                    [j, j],     # x-axis (column index)
                    [i, i],     # y-axis (row index)
                    [0, td[i, j]],  # z-axis from 0 to delay value
                    marker='o',
                    color='b'
                )

        ax.set_xlabel("Element index (y-direction)")
        ax.set_ylabel("Element index (x-direction)")
        ax.set_zlabel("Time Delay (µs)")
        ax.set_title(plot_title)

        # Use CLI parameters for camera viewing angle
        ax.view_init(elev=args.elev, azim=args.azim)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

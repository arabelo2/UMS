#!/usr/bin/env python3
"""
Module: mps_array_modeling_interface.py
Layer: Interface

Provides a CLI to compute the normalized pressure field for a 2D array of rectangular elements.
It uses delay_laws3D, ps_3Dv, and discrete_windows modules.

Default values:
  lx        = 0.15 mm
  ly        = 0.15 mm
  gx        = 0.05 mm
  gy        = 0.05 mm
  f         = 5 MHz
  c         = 1480 m/s
  L1        = 11
  L2        = 11
  theta     = 20 deg
  phi       = 0 deg
  F         = inf (steering-only)
  ampx_type = 'rect'
  ampy_type = 'rect'
  xs        = linspace(-15,15,300)
  zs        = linspace(1,20,200)
  y         = 0

Example usage:
  1) Steering Only (F = inf):
     python interface/mps_array_modeling_interface.py --lx 0.15 --ly 0.15 --gx 0.05 --gy 0.05 --f 5 --c 1480 --L1 11 --L2 11 --theta 20 --phi 0 --F inf --ampx_type rect --ampy_type rect --plot Y

  2) Steering + Focusing (F finite):
     python interface/mps_array_modeling_interface.py --lx 0.15 --ly 0.15 --gx 0.05 --gy 0.05 --f 5 --c 1480 --L1 11 --L2 11 --theta 20 --phi 0 --F 15 --ampx_type rect --ampy_type rect --plot Y
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.mps_array_modeling_service import run_mps_array_modeling_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure field for a 2D array using mps_array_modeling.",
        epilog="""Example usage:
  1) Steering Only (F = inf):
     python interface/mps_array_modeling_interface.py --lx 0.15 --ly 0.15 --gx 0.05 --gy 0.05 --f 5 --c 1480 --L1 11 --L2 11 --theta 20 --phi 0 --F inf --ampx_type rect --ampy_type rect --plot Y

  2) Steering + Focusing (F finite):
     python interface/mps_array_modeling_interface.py --lx 0.15 --ly 0.15 --gx 0.05 --gy 0.05 --f 5 --c 1480 --L1 11 --L2 11 --theta 20 --phi 0 --F 15 --ampx_type rect --ampy_type rect --plot Y
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--lx", type=safe_float, default=0.15,
                        help="Element length in x-direction (mm). Default=0.15.")
    parser.add_argument("--ly", type=safe_float, default=0.15,
                        help="Element length in y-direction (mm). Default=0.15.")
    parser.add_argument("--gx", type=safe_float, default=0.05,
                        help="Gap length in x-direction (mm). Default=0.05.")
    parser.add_argument("--gy", type=safe_float, default=0.05,
                        help="Gap length in y-direction (mm). Default=0.05.")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default=5.")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default=1480.")
    parser.add_argument("--L1", type=int, default=11,
                        help="Number of elements in x-direction. Default=11.")
    parser.add_argument("--L2", type=int, default=11,
                        help="Number of elements in y-direction. Default=11.")
    parser.add_argument("--theta", type=safe_float, default=20.0,
                        help="Steering angle theta (degrees). Default=20.")
    parser.add_argument("--phi", type=safe_float, default=0.0,
                        help="Steering angle phi (degrees). Default=0.")
    parser.add_argument("--F", type=safe_float, default=float('inf'),
                        help="Focal distance in mm (use inf for steering-only, finite for focusing). Default=inf.")
    parser.add_argument("--ampx_type", type=str, default="rect",
                        help="Window type for x-direction amplitudes. Default=rect.")
    parser.add_argument("--ampy_type", type=str, default="rect",
                        help="Window type for y-direction amplitudes. Default=rect.")
    parser.add_argument("--xs", type=str, default="-15,15,300",
                        help="Comma-separated values for xs: start, stop, num_points. Default='-15,15,300'.")
    parser.add_argument("--zs", type=str, default="1,20,200",
                        help="Comma-separated values for zs: start, stop, num_points. Default='1,20,200'.")
    parser.add_argument("--y", type=safe_float, default=0.0,
                        help="Fixed y-coordinate for evaluation. Default=0.")
    parser.add_argument("--plot", type=str, choices=["Y", "N", "y", "n"], default="Y",
                        help="Display pressure field plot: 'Y' for yes, 'N' for no. Default=Y.")
    
    args = parser.parse_args()
    
    # Parse xs and zs arrays using the helper.
    xs = parse_array(args.xs)
    zs = parse_array(args.zs)
    
    try:
        # Pass F to the service function (even if the underlying API expects it as Fl)
        p, xs_out, zs_out = run_mps_array_modeling_service(
            args.lx, args.ly, args.gx, args.gy,
            args.f, args.c, args.L1, args.L2,
            args.theta, args.phi, args.F,
            args.ampx_type, args.ampy_type,
            xs, zs, args.y
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save the magnitude of the pressure field to a file.
    outfile = "mps_array_modeling_output.txt"
    np.savetxt(outfile, np.abs(p), fmt="%.6f")
    print(f"Pressure field magnitude saved to {outfile}")
    
    # Plot the pressure field if requested.
    if args.plot.upper() == "Y":
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(p),
                   cmap="jet", extent=[xs_out.min(), xs_out.max(), zs_out.max(), zs_out.min()],
                   aspect="auto")
        plt.colorbar(label="Pressure Magnitude")
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        # Determine plot title based on F: infinite means steering-only, otherwise focusing.
        if math.isinf(args.F):
            plt.title("MPS Array Modeling - Steering Only")
        else:
            plt.title("MPS Array Modeling - Steering + Focusing")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

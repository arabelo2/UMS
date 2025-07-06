#!/usr/bin/env python3
"""
Module: mps_array_modeling_interface.py
Layer: Interface

Provides a CLI to compute the normalized pressure field for a 2D array of rectangular elements 
radiating waves in a fluid using the ps_3Dv algorithm. It utilizes the
mps_array_modeling_service to compute the field, then saves and optionally plots the results.

Default values:
  lx        = 0.15 mm
  ly        = 0.15 mm
  gx        = 0.05 mm
  gy        = 0.05 mm
  f         = 5 MHz
  c         = 1480 m/s
  L1        = 11
  L2        = 11
  theta     = 20 deg (steering angle in theta direction)
  phi       = 0 deg
  F         = inf mm (focal distance; use inf for steering-only)
  ampx_type = 'rect' (apodization type in x-direction)
  ampy_type = 'rect' (apodization type in y-direction)
  xs        = linspace(-15,15,300)
  zs        = linspace(1,20,200)
  y         = 0 (fixed y-coordinate)
  plot      = 'Y' (plot the pressure field)

Example usage:
  python interface/mps_array_modeling_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --c=1480 \
         --L1=11 --L2=11 --theta=20 --phi=0 --F=inf --ampx_type=rect --ampy_type=rect \
         --xs="-15,15,300" --zs="1,20,200" --y=0 --plot=Y
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.mps_array_modeling_service import run_mps_array_modeling_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute the normalized pressure field for a 2D array of rectangular elements.",
        epilog=(
            "Example usage:\n"
            "  python interface/mps_array_modeling_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --c=1480 "
            "--L1=11 --L2=11 --theta=20 --phi=0 --F=inf --ampx_type=rect --ampy_type=rect --xs=\"-15,15,300\" "
            "--zs=\"1,20,200\" --y=0 --plot=Y"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--lx", type=safe_float, default=0.15,
                        help="Element length in x-direction (mm). Default: 0.15")
    parser.add_argument("--ly", type=safe_float, default=0.15,
                        help="Element length in y-direction (mm). Default: 0.15")
    parser.add_argument("--gx", type=safe_float, default=0.05,
                        help="Gap length in x-direction (mm). Default: 0.05")
    parser.add_argument("--gy", type=safe_float, default=0.05,
                        help="Gap length in y-direction (mm). Default: 0.05")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default: 5")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default: 1480")
    parser.add_argument("--L1", type=int, default=11,
                        help="Number of elements in x-direction. Default: 11")
    parser.add_argument("--L2", type=int, default=11,
                        help="Number of elements in y-direction. Default: 11")
    parser.add_argument("--theta", type=safe_float, default=20.0,
                        help="Steering angle in theta direction (deg). Default: 20")
    parser.add_argument("--phi", type=safe_float, default=0.0,
                        help="Steering angle in phi direction (deg). Default: 0")
    parser.add_argument("--F", type=safe_float, default=float('inf'),
                        help="Focal distance (mm). Use inf for steering-only. Default: inf")
    parser.add_argument("--ampx_type", type=str, default="rect",
                        choices=["rect", "cos", "Han", "Ham", "Blk", "tri"],
                        help="Window type in x-direction. Default: rect")
    parser.add_argument("--ampy_type", type=str, default="rect",
                        choices=["rect", "cos", "Han", "Ham", "Blk", "tri"],
                        help="Window type in y-direction. Default: rect")
    parser.add_argument("--xs", type=str, default="-15,15,300",
                        help="x-coordinates as a comma-separated list or start,stop,num_points. Default: \"-15,15,300\"")
    parser.add_argument("--zs", type=str, default="1,20,200",
                        help="z-coordinates as a comma-separated list or start,stop,num_points. Default: \"1,20,200\"")
    parser.add_argument("--y", type=safe_float, default=0.0,
                        help="Fixed y-coordinate for evaluation. Default: 0")
    parser.add_argument("--plot", type=lambda s: s.upper(), choices=["Y", "N"], default="Y",
                        help="Plot the pressure field? (Y/N). Default: Y")
    
    args = parser.parse_args()
    
    # Process xs and zs arrays using the helper function.
    try:
        xs_vals = parse_array(args.xs)
    except Exception as e:
        parser.error(str(e))
    try:
        zs_vals = parse_array(args.zs)
    except Exception as e:
        parser.error(str(e))
    
    # Call the application service to compute the pressure field.
    p, xs_result, zs_result = run_mps_array_modeling_service(
         args.lx, args.ly, args.gx, args.gy,
         args.f, args.c, args.L1, args.L2,
         args.theta, args.phi, args.F,
         args.ampx_type, args.ampy_type,
         xs_vals, zs_vals, args.y
    )
    
    # Save the pressure field to a text file.
    outfile = "mps_array_model_output.txt"
    with open(outfile, "w") as f:
         for i in range(p.shape[0]):
             for j in range(p.shape[1]):
                 f.write(f"{p[i, j].real:.6f}+{p[i, j].imag:.6f}j\n")
    print(f"Pressure field saved to {outfile}")
    
    # Optionally, plot the pressure magnitude.
    if args.plot == "Y":
         plt.figure(figsize=(10, 6))
         plt.imshow(np.abs(p), cmap="gray",
                    extent=[xs_result.min(), xs_result.max(), zs_result.max(), zs_result.min()],
                    aspect='auto')
         plt.xlabel("x (mm)", fontsize=16)
         plt.ylabel("z (mm)", fontsize=16)
         if np.isinf(args.F):
             title = f"MPS Steered Beam\n($\\Theta$={args.theta}°, $\\Phi$={args.phi}°, F=∞, L$_1$={args.L1}, L$_2$={args.L2}, f={args.f} MHz, Window$_x$={args.ampx_type}, Window$_y$={args.ampy_type})"
         else:
             title = f"MPS Steered + Focused Beam\n($\\Theta$={args.theta}°, $\\Phi$={args.phi}°, F={args.F} mm, L$_1$={args.L1}, L$_2$={args.L2}, f={args.f} MHz, Window$_x$={args.ampx_type}, Window$_y$={args.ampy_type})"
         plt.title(title, fontsize=18, linespacing=1.2)
         cbar = plt.colorbar()
         cbar.set_label("Normalized Pressure Magnitude", fontsize=16, linespacing=1.2)
         cbar.ax.tick_params(labelsize=14)
         plt.tick_params(axis='both', labelsize=16)
         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
         plt.minorticks_on()
         plt.tight_layout()
         plt.show()

if __name__ == "__main__":
    main()

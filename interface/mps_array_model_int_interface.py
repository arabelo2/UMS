#!/usr/bin/env python3
"""
Module: mps_array_model_int_interface.py
Layer: Interface

Provides a CLI to compute the normalized velocity field for an array of 1-D elements 
radiating waves through a fluid/solid interface using the ps_3Dint algorithm.
It utilizes the mps_array_model_int service to compute the field, then saves and optionally plots the results.

Default values:
  lx        = 0.15 mm
  ly        = 0.15 mm
  gx        = 0.05 mm
  gy        = 0.05 mm
  f         = 5 MHz
  d1        = 1.0 (density, medium one)
  c1        = 1480 m/s (compressional wave speed, medium one)
  d2        = 7.9 (density, medium two)
  c2        = 5900 m/s (compressional wave speed, medium two)
  cs2       = 3200 m/s (shear wave speed, medium two)
  type      = 'p'  (wave type for medium two)
  L1        = 11
  L2        = 11
  angt      = 10.217 deg  (array angle)
  Dt0       = 50.8 mm     (height of array center above interface)
  theta20   = 20 deg      (steering angle in theta direction)
  phi       = 0 deg       (steering angle in phi direction)
  DF        = inf         (focal distance; use inf for steering-only)
  ampx_type = 'rect'      (apodization in x-direction)
  ampy_type = 'rect'      (apodization in y-direction)
  xs        = linspace(-5,20,100)
  zs        = linspace(1,20,100)
  y         = 0

Example usage:
  1) With plot:
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 --theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y=0 --plot=y
     
  2) Without plot:
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 --theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y=0 --plot=n
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.mps_array_model_int_service import run_mps_array_model_int_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute the normalized velocity field for an array at a fluid/solid interface.",
        epilog=(
            "Example usage:\n"
            "  1) With plot:\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 --theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" --y=0 --plot=y\n\n"
            "  2) Without plot:\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 --d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 --theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" --y=0 --plot=n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--lx", type=safe_float, default=0.15, help="Element length in x-direction (mm). Default: 0.15")
    parser.add_argument("--ly", type=safe_float, default=0.15, help="Element length in y-direction (mm). Default: 0.15")
    parser.add_argument("--gx", type=safe_float, default=0.05, help="Gap length in x-direction (mm). Default: 0.05")
    parser.add_argument("--gy", type=safe_float, default=0.05, help="Gap length in y-direction (mm). Default: 0.05")
    parser.add_argument("--f", type=safe_float, default=5.0, help="Frequency (MHz). Default: 5")
    parser.add_argument("--d1", type=safe_float, default=1.0, help="Density of medium one. Default: 1.0")
    parser.add_argument("--c1", type=safe_float, default=1480.0, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--d2", type=safe_float, default=7.9, help="Density of medium two. Default: 7.9")
    parser.add_argument("--c2", type=safe_float, default=5900.0, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--cs2", type=safe_float, default=3200.0, help="Shear wave speed in second medium (m/s). Default: 3200")
    parser.add_argument("--type", type=str, default="p", choices=["p", "s"], help="Wave type for medium two ('p' or 's'). Default: p")
    parser.add_argument("--L1", type=int, default=11, help="Number of elements in x-direction. Default: 11")
    parser.add_argument("--L2", type=int, default=11, help="Number of elements in y-direction. Default: 11")
    parser.add_argument("--angt", type=safe_float, default=10.217, help="Array angle with interface (deg). Default: 10.217")
    parser.add_argument("--Dt0", type=safe_float, default=50.8, help="Height of array center above interface (mm). Default: 50.8")
    parser.add_argument("--theta20", type=safe_float, default=20.0, help="Steering angle in theta direction (deg). Default: 20")
    parser.add_argument("--phi", type=safe_float, default=0.0, help="Steering angle in phi direction (deg). Default: 0")
    parser.add_argument("--DF", type=safe_float, default=float('inf'), help="Focal distance (mm). Use inf for steering-only. Default: inf")
    parser.add_argument("--ampx_type", type=str, default="rect", choices=["rect", "cos", "Han", "Ham", "Blk", "tri"], help="Window type in x-direction. Default: rect")
    parser.add_argument("--ampy_type", type=str, default="rect", choices=["rect", "cos", "Han", "Ham", "Blk", "tri"], help="Window type in y-direction. Default: rect")
    parser.add_argument("--xs", type=str, default="-5,20,100", help="x-coordinates as comma-separated list or start,stop,num_points. Default: \"-5,20,100\"")
    parser.add_argument("--zs", type=str, default="1,20,100", help="z-coordinates as comma-separated list or start,stop,num_points. Default: \"1,20,100\"")
    parser.add_argument("--y", type=safe_float, default=0.0, help="Fixed y-coordinate for evaluation. Default: 0")
    parser.add_argument("--plot", type=lambda s: s.lower(), choices=["y", "n"], default="y", help="Plot the pressure field? (y/n). Default: y")
    parser.add_argument("--z_scale", type=safe_float, default=10.0, help="Scale factor for z-axis (delay values) in stem plot. Default: 10")
    
    args = parser.parse_args()
    
    # Process xs and zs arrays using the helper.
    if args.xs is None or args.xs.strip() == "":
        x_vals = None
    else:
        try:
            x_vals = parse_array(args.xs)
        except Exception as e:
            parser.error(str(e))
    
    if args.zs is None or args.zs.strip() == "":
        z_vals = None
    else:
        try:
            z_vals = parse_array(args.zs)
        except Exception as e:
            parser.error(str(e))
    
    result = run_mps_array_model_int_service(
         args.lx, args.ly, args.gx, args.gy,
         args.f, args.d1, args.c1, args.d2, args.c2, args.cs2, args.type,
         args.L1, args.L2, args.angt, args.Dt0,
         args.theta20, args.phi, args.DF,
         args.ampx_type, args.ampy_type,
         x_vals, z_vals, args.y
    )
    p = result['p']
    x = result['x']
    z = result['z']
    
    outfile = "mps_array_model_int_output.txt"
    with open(outfile, "w") as f:
         for i in range(p.shape[0]):
             for j in range(p.shape[1]):
                 f.write(f"{p[i, j].real:.6f}+{p[i, j].imag:.6f}j\n")
    print(f"Results saved to {outfile}")
    
    if args.plot == "y":
         plt.figure(figsize=(10, 6))
         plt.imshow(np.abs(p), cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
         plt.xlabel("x (mm)")
         plt.ylabel("z (mm)")
         # Determine medium type based on density thresholds (e.g., threshold = 2.0)
         medium1 = "Fluid" if args.d1 < 2.0 else "Solid"
         medium2 = "Fluid" if args.d2 < 2.0 else "Solid"
         if args.d1 == args.d2 and args.c1 == args.c2:
             plot_title = f"MLS Array Modeling Pressure Field for {medium1}"
         else:
             plot_title = f"MLS Array Modeling Pressure Field at {medium1}/{medium2} Interface"
         plt.title(plot_title)
         plt.colorbar(label="Pressure Magnitude")
         plt.tight_layout()
         plt.show()

if __name__ == "__main__":
    main()

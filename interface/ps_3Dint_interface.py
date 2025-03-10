#!/usr/bin/env python3
# interface/ps_3Dint_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Module: ps_3Dint_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing the velocity components (vx, vy, vz)
using the ps_3Dint algorithm. This algorithm computes the normalized velocity components
for a rectangular array element radiating waves through a fluid-solid interface.

Default values:
    lx   = 6 (mm)
    ly   = 12 (mm)
    f    = 5 (MHz)
    mat  = [1, 1480, 7.9, 5900, 3200, 'p']  (material properties: d1, cp1, d2, cp2, cs2, wave_type)
    angt = 10.217 (degrees)
    Dt0  = 50.8 (mm)
    x    = linspace(0, 30, 100) (mm)
    z    = linspace(1, 20, 100) (mm)
    y    = 0 (fixed value)

Example usage:
    python interface/ps_3Dint_interface.py --lx 6 --ly 12 --f 5 --mat "1,1480,7.9,5900,3200,p" \
       --angt 10.217 --Dt0 50.8 --x="0,30,100" --z="1,20,100" --outfile "velocity_output.txt" --plotfile "plot.png"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.ps_3Dint_service import run_ps_3Dint_service
from interface.cli_utils import safe_eval, safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute velocity components (vx, vy, vz) using the ps_3Dint algorithm.",
        epilog=(
            "Example usage:\n"
            "  python interface/ps_3Dint_interface.py --lx 6 --ly 12 --f 5 --mat \"1,1480,7.9,5900,3200,p\" \\\n"
            "       --angt 10.217 --Dt0 50.8 --x=\"0,30,100\" --z=\"1,20,100\" --outfile \"velocity_output.txt\" --plotfile \"plot.png\"\n\n"
            "Defaults:\n"
            "  lx   = 6 (mm)\n"
            "  ly   = 12 (mm)\n"
            "  f    = 5 (MHz)\n"
            "  mat  = [1, 1480, 7.9, 5900, 3200, 'p']\n"
            "  angt = 10.217 (degrees)\n"
            "  Dt0  = 50.8 (mm)\n"
            "  x    = linspace(0, 30, 100) (mm)\n"
            "  z    = linspace(1, 20, 100) (mm)\n"
            "  y    = 0 (fixed value)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Define default values
    parser.add_argument("--lx", type=safe_float, default=6.0,
                        help="Length of the array element in the x-direction (mm). Default: 6.0")
    parser.add_argument("--ly", type=safe_float, default=12.0,
                        help="Length of the array element in the y-direction (mm). Default: 12.0")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--mat", type=str, default="1,1480,7.9,5900,3200,p",
                        help="Material properties as comma-separated values: d1,cp1,d2,cp2,cs2,wave_type. Default: '1,1480,7.9,5900,3200,p'")
    parser.add_argument("--angt", type=safe_float, default=10.217,
                        help="Angle of the array with respect to the interface (degrees). Default: 10.217")
    parser.add_argument("--Dt0", type=safe_float, default=50.8,
                        help="Distance from the array center to the interface (mm). Default: 50.8")
    parser.add_argument("--x", type=str, default="0,30,100",
                        help="x-coordinates (mm). If exactly three numbers are provided, they are interpreted as start, stop, and number of points. Default: '0,30,100'")
    parser.add_argument("--z", type=str, default="1,20,100",
                        help="z-coordinates (mm). If exactly three numbers are provided, they are interpreted as start, stop, and number of points. Default: '1,20,100'")
    parser.add_argument("--y", type=safe_float, default=0.0,
                        help="y-coordinate (fixed value). Default: 0.0")
    parser.add_argument("--outfile", type=str, default="velocity_output.txt",
                        help="Output file to save the velocity magnitude matrix. Default: velocity_output.txt")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="If provided, the plot will be saved to this file (e.g., 'plot.png').")

    args = parser.parse_args()

    # Parse material properties
    try:
        mat_values = [float(item.strip()) if item.strip().replace('.', '', 1).isdigit() else item.strip() for item in args.mat.split(",")]
        if len(mat_values) != 6:
            raise ValueError
    except ValueError:
        parser.error("mat must contain exactly six comma-separated values: d1,cp1,d2,cp2,cs2,wave_type (e.g., '1,1480,7.9,5900,3200,p').")

    # Parse x and z coordinates
    try:
        x = parse_array(args.x)
        z = parse_array(args.z)
    except Exception as e:
        parser.error(str(e))

    # Create meshgrid for 2D simulation
    xx, zz = np.meshgrid(x, z)

    # Call the service layer
    vx, vy, vz = run_ps_3Dint_service(args.lx, args.ly, args.f, mat_values, 0, 0, args.angt, args.Dt0, xx, args.y, zz)

    # Compute magnitude of velocity
    v = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)

    # Plot the result
    plt.imshow(v, cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.title('Velocity Magnitude in the Second Medium')

    # Save the plot if a filename is provided
    if args.plotfile:
        plt.savefig(args.plotfile, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.plotfile}")

    plt.show()

    # Save the velocity magnitude matrix to a file
    try:
        with open(args.outfile, "w") as f:
            for row in v:
                formatted_row = "\t".join("%0.16f" % val for val in row)
                f.write(formatted_row + "\n")
        print(f"Velocity magnitude matrix saved in {args.outfile}")
    except Exception as e:
        print("Error saving the output file:", e)

if __name__ == "__main__":
    main()
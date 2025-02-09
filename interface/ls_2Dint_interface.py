# interface/ls_2Dint_interface.py

r"""
Module: ls_2Dint_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing LS2D pressure.
Default values are provided so that if the user omits x and z, a default 2-D grid is used.

By default, if x and z are not provided, the program creates a 2D grid as follows:

    b = 3;
    f = 5;
    mat = [1,1480,7.9,5900];
    angt = 10.217;
    Dt0 = 50.8;
    x = linspace(0,25,200);
    z = linspace(1,25,200);
    [xx, zz] = meshgrid(x,z);
    p = ls_2Dint(b, f, mat, 0, angt, Dt0, xx, zz);
    imagesc(x, z, abs(p));

Example usage:
    py ./interface/ls_2Dint_interface.py --b 3 --f 5 --mat "1,1480,7.9,5900" --e 0 --angt 10.217 --Dt0 50.8
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dint_service import LS2DInterfaceService

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure using the LS2D model."
    )
    parser.add_argument("--b", type=float, nargs="?", default=3.0,
                        help="Half-length of the source (mm). Default: 3.0")
    parser.add_argument("--f", type=float, nargs="?", default=5.0,
                        help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--mat", type=str, nargs="?", default="1,1480,7.9,5900",
                        help="Material properties as comma-separated values: d1,c1,d2,c2. Default: '1,1480,7.9,5900'")
    parser.add_argument("--e", type=float, nargs="?", default=0.0,
                        help="Offset of the element from the array center (mm). Default: 0.0")
    parser.add_argument("--angt", type=float, nargs="?", default=10.217,
                        help="Angle of the array with respect to the x-axis (degrees). Default: 10.217")
    parser.add_argument("--Dt0", type=float, nargs="?", default=50.8,
                        help="Distance of the array center above the interface (mm). Default: 50.8")
    parser.add_argument("--x", type=str, nargs="?", default=None,
                        help="x-coordinates: either a single number or comma-separated list. If omitted, a default grid is used.")
    parser.add_argument("--z", type=str, nargs="?", default=None,
                        help="z-coordinates: either a single number or comma-separated list. If omitted, a default grid is used.")
    parser.add_argument("--Nopt", type=int, nargs="?", default=None,
                        help="(Optional) Number of segments to use. Default: computed automatically.")
    parser.add_argument("--outfile", type=str, nargs="?", default="pressure_output.txt",
                        help="Output file to save the pressure p. Default: pressure_output.txt")
    parser.add_argument("--plotfile", type=str, nargs="?", default=None,
                        help="(Optional) If provided, the plot will be saved to this file (e.g., 'plot.png').")
    
    args = parser.parse_args()

    # Parse the material properties.
    try:
        mat_values = [float(item.strip()) for item in args.mat.split(",")]
        if len(mat_values) != 4:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("mat must contain exactly four comma-separated numbers.")

    # Determine x and z:
    if args.x is None or args.z is None:        
        x = np.linspace(0, 25, 200)
        z = np.linspace(1, 25, 200)
        xx, zz = np.meshgrid(x, z)
    else:
        try:
            x_vals = [float(item.strip()) for item in args.x.split(",")]
            z_vals = [float(item.strip()) for item in args.z.split(",")]
        except Exception:
            raise argparse.ArgumentTypeError("x and z must be numbers or comma-separated lists of numbers.")
        if len(x_vals) == 1 and len(z_vals) == 1:
            xx, zz = x_vals[0], z_vals[0]
        else:
            xx, zz = np.meshgrid(np.array(x_vals), np.array(z_vals))

    service = LS2DInterfaceService(args.b, args.f, mat_values, args.e, args.angt, args.Dt0, xx, zz, args.Nopt)
    p = service.calculate()
      
    print("Normalized Pressure (p):", p)
    
    # Save the pressure matrix to a text file with custom formatting.
    with open(args.outfile, "w") as f:
        # Iterate over each row in the pressure matrix.
        for row in p:
            # Format each complex number as (real+imagj) with 16 decimal places.
            formatted_row = "\t".join("%0.16f%+0.16fj" % (val.real, val.imag) for val in row)
            f.write(formatted_row + "\n")
    print(f"Pressure matrix saved in {args.outfile}")
    
    # Then continue with plotting:
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(p), extent=[np.min(xx), np.max(xx), np.min(zz), np.max(zz)],
               origin="lower", aspect="auto", cmap="jet")
    plt.colorbar(label="Normalized Pressure |p|")
    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("2D Interface Pressure Distribution")
    plt.show()

if __name__ == "__main__":
    main()

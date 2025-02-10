# interface/ls_2Dint_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
"""
Module: ls_2Dint_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing the normalized pressure 
using the LS2D (2-D Interface) model. This model computes the pressure for a 
single source radiating waves across a fluid-fluid interface using a Rayleigh-Sommerfeld 
integral with ray theory for cylindrical wave propagation.

If x and z are not provided for 1D simulation, the program uses default values:
    b     = 3 (mm)
    f     = 5 (MHz)
    mat   = [1, 1480, 7.9, 5900]  (material properties: d1, c1, d2, c2)
    angt  = 10.217 (degrees)
    Dt0   = 50.8 (mm)
    x1    = 0 (fixed value)
    z1    = linspace(1,100,200) (mm)
For 2D simulation, if x2 and z2 are not provided, defaults are:
    x2 = linspace(0,25,200)
    z2 = linspace(1,25,200)

Example usage:
    py interface/ls_2Dint_interface.py --b 3 --f 5 --c 1480 --mat "1,1480,7.9,5900" --e 0 \
       --angt 10.217 --Dt0 50.8 --x2="0,25,200" --z2="1,25,200" --Nopt 20 \
       --outfile "pressure_output.txt" --plotfile "plot.png"
       
Note: For 1D simulation, it is recommended that x is a fixed value and z is provided as a vector.
For negative values in --x, --z, --x2, or --z2, enclose the argument in quotes (e.g., --x2="0,25,200").
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dint_service import run_ls_2Dint_service
from interface.cli_utils import safe_eval, safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure using the LS2D (2-D Interface) model.",
        epilog=(
            "Example usage:\n"
            "  python interface/ls_2Dint_interface.py --b 3 --f 5 --c 1480 --mat \"1,1480,7.9,5900\" --e 0 \\\n"
            "       --angt 10.217 --Dt0 50.8 --x2=\"0,25,20\" --z2=\"1,25,200\" --Nopt 20 \n\n"
            "Defaults:\n"
            "  For 1D simulation: x is taken as a fixed value and z as a vector.\n"
            "    If --x and --z are omitted, x defaults to 0 and z to linspace(1,100,200).\n"
            "  For 2D simulation: x2 defaults to linspace(0,25,200) and z2 to linspace(1,25,200).\n"
            "The --Nopt option specifies the number of segments; if not provided, it is computed automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 1D simulation parameters.
    parser.add_argument("--b", type=safe_float, default=3.0,
                        help="Half-length of the source (mm). Default: 3.0")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default: 1480.0")
    parser.add_argument("--mat", type=str, default="1,1480,7.9,5900",
                        help="Material properties as comma-separated values: d1,c1,d2,c2. Default: '1,1480,7.9,5900'")
    parser.add_argument("--e", type=safe_float, default=0.0,
                        help="Offset of the element from the array center (mm). Default: 0.0")
    parser.add_argument("--angt", type=safe_float, default=10.217,
                        help="Angle of the array with respect to the x-axis (degrees). Default: 10.217")
    parser.add_argument("--Dt0", type=safe_float, default=50.8,
                        help="Distance from the array center to the interface (mm). Default: 50.8")
    parser.add_argument("--x", type=str, default=None,
                        help=("x-coordinate(s) for 1D simulation: either a single fixed number or a comma-separated list. "
                              "For 1D, it is recommended to supply a single value. If omitted, x defaults to 0."))
    parser.add_argument("--z", type=str, default=None,
                        help=("z-coordinate(s) for 1D simulation: if exactly one value is provided, a default grid is used; "
                              "if multiple values are provided, they are used as a vector. "
                              "If omitted, z defaults to linspace(5,80,200)."))
    parser.add_argument("--Nopt", type=int, default=None,
                        help="(Optional) Number of segments for numerical integration. Default: computed automatically.")
    
    # 2D simulation parameters.
    parser.add_argument("--x2", type=str, default=None,
                        help=("x-coordinate(s) for 2D simulation: either a single number or a comma-separated list. "
                              "If exactly three numbers are provided, they are interpreted as start, stop, and number of points. "
                              "Default: linspace(0,25,200)."))
    parser.add_argument("--z2", type=str, default=None,
                        help=("z-coordinate(s) for 2D simulation: either a single number or a comma-separated list. "
                              "If exactly three numbers are provided, they are interpreted as start, stop, and number of points. "
                              "Default: linspace(1,25,200)."))
    
    # Plot mode.
    parser.add_argument("--plot-mode", type=str, choices=["both", "1D", "2D"], default="2D",
                        help="Plot mode: 'both' for both 1D and 2D, '1D' for 1D only, or '2D' for 2D only. Default: both.")
    
    # Output files.
    parser.add_argument("--outfile", type=str, default="pressure_output.txt",
                        help="Output file to save the pressure matrix. Default: pressure_output.txt")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="If provided, the plot will be saved to this file (e.g., 'plot.png').")
    
    args = parser.parse_args()
    
    # Parse material properties.
    try:
        mat_values = [float(item.strip()) for item in args.mat.split(",")]
        if len(mat_values) != 4:
            raise ValueError
    except ValueError:
        parser.error("mat must contain exactly four comma-separated numbers (e.g., '1,1480,7.9,5900').")
    
    # Process 1D simulation x and z coordinates.
    if args.x is None or args.z is None:
        # If either is omitted, default: x = 0 (fixed) and z = linspace(5,80,200)
        x1 = 0
        z1 = np.linspace(5, 80, 200)
    else:
        try:
            x_vals = [float(item.strip()) for item in args.x.split(",")]
            z_vals = [float(item.strip()) for item in args.z.split(",")]
        except Exception:
            parser.error("x and z must be numbers or comma-separated lists of numbers.")
        # If x is a single value and z is a vector, use that.
        if len(x_vals) == 1 and len(z_vals) >= 1:
            x1 = x_vals[0]
            z1 = np.array(z_vals) if len(z_vals) > 1 else np.linspace(5, 80, 200)
        elif len(z_vals) == 1 and len(x_vals) >= 1:
            # If z is a single value and x is a vector, assume x should be a fixed value.
            # We choose the first value as the fixed coordinate.
            x1 = np.array(x_vals) if len(x_vals) > 1 else x_vals[0]
            z1 = z_vals[0]
        elif len(x_vals) > 1 and len(z_vals) > 1:
            # Both provided as vectors: for 1D simulation we assume x should be fixed.
            print("Warning: Both x and z provided as vectors for 1D simulation. Using first x value as fixed coordinate.")
            x1 = x_vals[0]
            z1 = np.array(z_vals)
        else:
            # Both are scalars.
            x1 = x_vals[0]
            z1 = np.linspace(5, 50, 200)
    
    # Process 2D simulation coordinates.
    if args.x2 is None:
        x2 = np.linspace(0, 25, 200)
    else:
        try:
            x2 = parse_array(args.x2)
        except Exception as e:
            parser.error(str(e))
    if args.z2 is None:
        z2 = np.linspace(1, 25, 200)
    else:
        try:
            z2 = parse_array(args.z2)
        except Exception as e:
            parser.error(str(e))
    
    # Create meshgrid for 2D simulation if needed.
    if args.plot_mode in ["both", "2D"]:
        xx, zz = np.meshgrid(x2, z2)
    else:
        xx, zz = None, None
    
    # Call the service layer.
    p1 = None
    if args.plot_mode in ["both", "1D"]:
        p1 = run_ls_2Dint_service(args.b, args.f, args.c, args.e, mat_values, args.angt, args.Dt0, x1, z1, args.Nopt)
        if isinstance(p1, np.ndarray) and p1.ndim > 0 and p1.shape[0] == 1:
            p1 = np.squeeze(p1, axis=0)
    
    p2 = None
    if args.plot_mode in ["both", "2D"]:
        p2 = run_ls_2Dint_service(args.b, args.f, args.c, args.e, mat_values, args.angt, args.Dt0, xx, zz, args.Nopt)
        if isinstance(p2, np.ndarray) and p2.ndim > 2 and p2.shape[0] == 1:
            p2 = np.squeeze(p2, axis=0)
    
    # Plotting.
    if args.plot_mode in ["both", "1D"] and p1 is not None:
        plt.figure(figsize=(8, 5))
        # If z1 is a vector, plot versus z1.
        if isinstance(z1, np.ndarray):
            plt.plot(z1, np.abs(p1), 'b-', lw=2)
            plt.xlabel("z (mm)")
        else:
            # Otherwise, plot versus a default linspace.
            default_z = np.linspace(5, 80, len(np.atleast_1d(p1)))
            plt.plot(default_z, np.abs(p1), 'b-', lw=2)
            plt.xlabel("z (mm)")
        plt.ylabel("Normalized Pressure Magnitude")
        plt.title("1D LS Simulation")
        plt.grid(True)
    
    if args.plot_mode in ["both", "2D"] and p2 is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(p2), cmap="jet", extent=[x2[0], x2[-1], z2[0], z2[-1]],
                   aspect="equal", origin="lower")
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("2D LS Simulation")
        plt.colorbar(label="Pressure Magnitude")
    
    plt.show()
    
    # Print output.
    print("Normalized Pressure (p):", p1 if p1 is not None else p2)
    
    # Save the pressure matrix to a file.
    try:
        with open(args.outfile, "w") as f:
            if p1 is not None:
                for row in p1:
                    formatted_row = "\t".join("%0.16f%+0.16fj" % (val.real, val.imag) for val in row)
                    f.write(formatted_row + "\n")
            elif p2 is not None:
                for row in p2:
                    formatted_row = "\t".join("%0.16f%+0.16fj" % (val.real, val.imag) for val in row)
                    f.write(formatted_row + "\n")
        print(f"Pressure matrix saved in {args.outfile}")
    except Exception as e:
        print("Error saving the output file:", e)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# interface/on_axis_foc2D_interface.py

"""
Module: on_axis_foc2D_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing the on-axis normalized pressure 
using the on_axis_foc2D model. This model computes the pressure for a 1-D focused piston 
element using a paraxial approximation expressed in terms of a Fresnel integral.

Default parameter values:
    b = 6         (Transducer half-length in mm)
    R = 100       (Focal length in mm)
    f = 5         (Frequency in MHz)
    c = 1480      (Wave speed in m/s)
    z = linspace(20,400,500) (On-axis distance in mm)

The pressure is computed as:
    p = on_axis_foc2D(b, R, f, c, z)
and the magnitude is plotted as |p| versus z.

Example usage:
    python interface/on_axis_foc2D_interface.py --b 6 --R 100 --f 5 --c 1480 --z "20,400,500"
    
Note: For the --z argument, if exactly three comma-separated numbers are provided, they are 
interpreted as start, stop, and number of points (e.g., "20,400,500" generates linspace(20,400,500)).
For negative values, enclose the argument in quotes.
"""

import sys
import os
# Ensure the project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.on_axis_foc2D_service import run_on_axis_foc2D_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute on-axis normalized pressure using the on_axis_foc2D model.",
        epilog=(
            "Example usage:\n"
            "  python interface/on_axis_foc2D_interface.py --b 6 --R 100 --f 5 --c 1480 --z \"20,400,500\"\n\n"
            "Defaults:\n"
            "  b = 6, R = 100, f = 5, c = 1480, z = linspace(20,400,500).\n"
            "The computed pressure p is plotted as |p| vs. z."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--b", type=safe_float, default=6.0,
                        help="Transducer half-length (mm). Default: 6.0")
    parser.add_argument("--R", type=safe_float, default=100.0,
                        help="Focal length (mm). Default: 100.0")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--c", type=safe_float, default=1480.0,
                        help="Wave speed (m/s). Default: 1480.0")
    parser.add_argument("--z", type=str, default=None,
                        help=("z-coordinate(s) (mm) where pressure is computed. "
                              "Provide a comma-separated list. If exactly three numbers are provided, "
                              "they are interpreted as start, stop, and number of points (e.g., \"20,400,500\"). "
                              "Default: linspace(20,400,500)."))
    
    parser.add_argument("--outfile", type=str, default="on_axis_pressure_output.txt",
                        help="Output file to save the pressure matrix. Default: on_axis_pressure_output.txt")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="If provided, the plot will be saved to this file (e.g., 'plot.png').")
    
    args = parser.parse_args()
    
    # Process z: if not provided, use default linspace(20,400,500)
    if args.z is None:
        z_vals = np.linspace(20, 400, 500)
    else:
        try:
            z_vals = parse_array(args.z)
        except Exception as e:
            parser.error(str(e))
    
    # Call the service layer.
    p = run_on_axis_foc2D_service(args.b, args.R, args.f, args.c, z_vals)
    
    # Print computed pressure.
    print("Normalized Pressure (p):", p)
    
    # Plot the pressure magnitude vs. z.
    plt.figure(figsize=(8, 5))
    plt.plot(z_vals, np.abs(p), 'b-', lw=2)
    plt.xlabel("z (mm)")
    plt.ylabel("Normalized Pressure Magnitude")
    plt.title("On-Axis Focused 2D Simulation")
    plt.grid(True)
    
    if args.plotfile:
        plt.savefig(args.plotfile)
        print(f"Plot saved in {args.plotfile}")
    else:
        plt.show()
    
    # Save the pressure matrix to a file.
    try:
        with open(args.outfile, "w") as f:
            # Check if p is scalar, 1D, or multi-dimensional.
            if np.isscalar(p) or (isinstance(p, np.ndarray) and p.ndim == 0):
                f.write("%0.16f%+0.16fj\n" % (p.real, p.imag))
            elif isinstance(p, np.ndarray) and p.ndim == 1:
                for val in p:
                    f.write("%0.16f%+0.16fj\n" % (val.real, val.imag))
            else:
                for row in p:
                    formatted_row = "\t".join("%0.16f%+0.16fj" % (val.real, val.imag) for val in row)
                    f.write(formatted_row + "\n")
        print(f"Pressure matrix saved in {args.outfile}")
    except Exception as e:
        print("Error saving the output file:", e)

if __name__ == "__main__":
    main()

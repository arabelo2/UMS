#!/usr/bin/env python3
# interface/fresnel_2D_interface.py

"""
Module: fresnel_2D_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing the normalized pressure
using the Fresnel 2D model. This model computes the pressure for a 1-D element
radiating into a fluid, using a Fresnel integral to approximate the pressure field.

Defaults (if x and z are not provided):
    b = 6 (mm)
    f = 5 (MHz)
    c = 1500 (m/s)
    z = 60 (mm, fixed value)
    x = linspace(-10,10,200) (mm)

The simulation is executed as:
    p = fresnel_2D(b, f, c, x, z)
and the result is plotted as:
    plot(x, abs(p))

Example usage:
    py ./fresnel_2D_interface.py --b 6 --f 5 --c 1500 --z 60 --x="-10,10,200" 
    [--outfile "pressure_output.txt" --plotfile "plot.png"]
    
Note: For negative values in --x, enclose the argument in quotes, e.g., --x="-10,10,200".
"""

import sys
import os
# Ensure the project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.fresnel_2D_service import run_fresnel_2D_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute Fresnel-based pressure field for a 1-D element in a fluid.",
        epilog=(
            "Example usage:\n"
            "  python fresnel_2D_interface.py --b 6 --f 5 --c 1500 --z 60 --x=\"-10,10,200\" \n\n"
            "Defaults:\n"
            "  b = 6, f = 5, c = 1500, z = 60, x = linspace(-10,10,200).\n"
            "The pressure field p is computed and plotted as |p| vs. x."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--b", type=safe_float, default=6.0,
                        help="Half-length of the element (mm). Default: 6.0")
    parser.add_argument("--f", type=safe_float, default=5.0,
                        help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--c", type=safe_float, default=1500.0,
                        help="Wave speed (m/s). Default: 1500.0")
    parser.add_argument("--z", type=safe_float, default=60.0,
                        help="z-coordinate (mm) at which pressure is computed. Default: 60.0")
    parser.add_argument("--x", type=str, default=None,
                        help=("x-coordinates (mm). Provide a comma-separated list. "
                              "If exactly three numbers are provided, they are interpreted as "
                              "start, stop, and number of points (e.g., \"-10,10,200\"). "
                              "Default: linspace(-10,10,200)."))
    
    parser.add_argument("--outfile", type=str, default="pressure_output.txt",
                        help="Output file to save the pressure matrix. Default: pressure_output.txt")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="If provided, the plot will be saved to this file (e.g., 'plot.png').")
    
    args = parser.parse_args()
    
    # Parse x-coordinates. If not provided, use default linspace.
    if args.x is None:
        x_vals = np.linspace(-10, 10, 200)
    else:
        try:
            x_vals = parse_array(args.x)
        except Exception as e:
            parser.error(str(e))
    
    # Compute Fresnel-based pressure using the application service.
    p = run_fresnel_2D_service(args.b, args.f, args.c, x_vals, args.z)
    
    # Print the computed pressure.
    print("Normalized Pressure (p):", p)
    
    # Save the pressure matrix to a file.
    try:
        with open(args.outfile, "w") as f:
            # Check dimensionality: if scalar or 1D, iterate accordingly.
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
    
    # Plot the pressure magnitude vs x if p is 1D.
    if isinstance(p, np.ndarray) and p.ndim == 1 and p.shape == x_vals.shape:
        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, np.abs(p), 'b-', lw=2)
        plt.xlabel("x (mm)")
        plt.ylabel("Normalized Pressure Magnitude")
        plt.title("Fresnel 2D Simulation")
        plt.grid(True)
        if args.plotfile:
            plt.savefig(args.plotfile)
            print(f"Plot saved in {args.plotfile}")
        else:
            plt.show()
    else:
        # If p is not 1D, show a generic plot.
        plt.figure(figsize=(8, 6))
        plt.plot(np.abs(p))
        plt.title("Pressure Magnitude")
        plt.show()

if __name__ == "__main__":
    main()

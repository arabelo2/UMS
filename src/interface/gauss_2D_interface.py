# interface/gauss_2D_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.gauss_2D_service import run_gauss_2D_service
from interface.cli_utils import parse_array, safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure using the Gauss 2D model.",
        epilog=(
            "Example usage:\n"
            "  python interface/gauss_2D_interface.py --b 6 --f 5 --c 1500 --z 60 --x1='-10,10,200' --x2='-10,10,40' --plot Y\n"
            "The default parameters simulate a Gaussian beam field and overlay Fresnel beam data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--b", type=safe_float, default=6, help="Half-length of the source (mm). Default: 6")
    parser.add_argument("--f", type=safe_float, default=5, help="Frequency (MHz). Default: 5")
    parser.add_argument("--c", type=safe_float, default=1500, help="Wave speed (m/s). Default: 1500")
    parser.add_argument("--z", type=safe_float, default=60, help="Z-coordinate (mm). Default: 60")
    parser.add_argument("--x1", type=str, default="-10,10,200", 
                        help="X1-coordinates for Fresnel as 'start,stop,num_points'. Default: '-10,10,200'")
    parser.add_argument("--x2", type=str, default="-10,10,40", 
                        help="X2-coordinates for Gauss 2D as 'start,stop,num_points'. Default: '-10,10,40'")
    parser.add_argument("--outfile", type=str, default="gauss_2D_output.txt", help="Output file name. Default: gauss_2D_output.txt")
    parser.add_argument("--plot", type=str, choices=["Y", "N"], default="Y", 
                        help="Display the plot: 'Y' for yes, 'N' for no. Default: 'Y'")

    args = parser.parse_args()

    # Parse x-coordinates
    x1_values = parse_array(args.x1)
    x2_values = parse_array(args.x2)

    # Run the Gauss 2D model
    p = run_gauss_2D_service(args.b, args.f, args.c, x2_values, args.z)

    # Save the results to a file
    with open(args.outfile, "w") as f:
        for val in p:
            f.write(f"{val.real:.6f}+{val.imag:.6f}j\n")
    print(f"Results saved to {args.outfile}")

    # Plot the results if requested
    if args.plot.upper() == "Y":
        plt.plot(x1_values, np.abs(run_gauss_2D_service(args.b, args.f, args.c, x1_values, args.z)), '-', label="Fresnel 2D")
        plt.plot(x2_values, np.abs(p), 'o', label="Gauss 2D")

        plt.xlabel("x (mm)")
        plt.ylabel("Normalized Pressure Magnitude")
        plt.title("Gauss 2D vs. Fresnel 2D Pressure Field")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

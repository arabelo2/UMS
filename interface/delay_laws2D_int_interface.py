#!/usr/bin/env python
# interface/delay_laws2D_int_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from interface.cli_utils import safe_float
from application.delay_laws2D_int_service import run_delay_laws2D_int_service

def main():
    parser = argparse.ArgumentParser(
        description="Compute delay laws for an array at a fluid/fluid interface using ls_2Dint.",
        epilog=(
            "Example usage:\n"
            "  1. Standard usage:\n"
            "     python interface/delay_laws2D_int_interface.py --M 32 --s 1.0 --angt 0 --ang20 30 "
            "--DT0 25.4 --DF inf --c1 1480 --c2 5900 --plt y --plot-type line\n"
            "     (Use DF=inf for steering-only, or a finite DF for steering and focusing.)\n"
            "     To use a stem plot instead of a line plot, add: --plot-type stem\n\n"
            "  2. Second example (matching MATLAB parameters):\n"
            "     python interface/delay_laws2D_int_interface.py --M 16 --s 0.5 --angt 5 --ang20 60 "
            "--DT0 10 --DF 10 --c1 1480 --c2 5900 --plt y --plot-type stem\n"
            "     (This example closely follows the MATLAB delay_laws2D_int usage.)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--M", type=int, default=32, help="Number of elements. Default: 32")
    parser.add_argument("--s", type=safe_float, default=1.0, help="Array pitch (mm). Default: 1.0")
    parser.add_argument("--angt", type=safe_float, default=0, help="Array angle with interface (degrees). Default: 0")
    parser.add_argument("--ang20", type=safe_float, default=30, help="Refracted angle in second medium (degrees). Default: 30")
    parser.add_argument("--DT0", type=safe_float, default=25.4, help="Height of array center above interface (mm). Default: 25.4")
    parser.add_argument("--DF", type=safe_float, default=float('inf'), help="Focal depth in second medium (mm). Default: inf")
    parser.add_argument("--c1", type=safe_float, default=1480, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--c2", type=safe_float, default=5900, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--plt", type=lambda s: s.lower(), choices=["y", "n"], default="y", help="Plot delay curves? y/n. Default: y")
    parser.add_argument("--plot-type", type=str, choices=["line", "stem"], default="line", help="Plot type: 'line' (default) or 'stem'.")

    args = parser.parse_args()

    # Compute delays using the service layer.
    delays = run_delay_laws2D_int_service(args.M, args.s, args.angt, args.ang20, args.DT0, args.DF, args.c1, args.c2, args.plt)
    
    print("Computed delays (in microseconds):")
    print(delays)
    
    # Optionally, plot the delays vs element index.
    if args.plt.lower() == 'y':
        element_indices = np.arange(1, args.M + 1)
        plt.figure(figsize=(8, 5))

        if args.plot_type == "stem":
            plt.stem(element_indices, delays, linefmt='b-', markerfmt='bo', basefmt=" ", label="Delay Laws")
        else:
            plt.plot(element_indices, delays, 'bo-', label="Delay Laws")

        plt.xlabel("Element Index")
        plt.ylabel("Delay (µs)")
        if np.isinf(args.DF):
            plot_title = "Delay Laws - Steering-only case"
        else:
            plot_title = "Delay Laws - Steering and focusing case"
        plt.title(plot_title)
        plt.grid(True)
        if any(line.get_label() != "_nolegend_" for line in plt.gca().get_lines()):
            plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

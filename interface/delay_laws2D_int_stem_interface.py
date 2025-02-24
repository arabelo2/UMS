#!/usr/bin/env python
# interface/delay_laws2D_int_stem_interface.py

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
        description="Compute delay laws for an array at a fluid/fluid interface using ls_2Dint, with stem plot visualization.",
        epilog=(
            "Example usage (with stem plot):\n"
            "  python interface/delay_laws2D_int_stem_interface.py --M 16 --s 0.5 --angt 5 --ang20 60 "
            "--DT0 10 --DF 10 --c1 1480 --c2 5900 --plt y\n\n"
            "Defaults:\n"
            "  M = 16, s = 0.5, angt = 5, ang20 = 60, DT0 = 10, DF = 10, c1 = 1480, c2 = 5900\n"
            "If DF=inf, the delay law is computed for steering-only; for a finite DF, steering and focusing are applied."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Default parameters for stem interface.
    parser.add_argument("--M", type=int, default=16, help="Number of elements. Default: 16")
    parser.add_argument("--s", type=safe_float, default=0.5, help="Array pitch (mm). Default: 0.5")
    parser.add_argument("--angt", type=safe_float, default=5, help="Array angle with interface (degrees). Default: 5")
    parser.add_argument("--ang20", type=safe_float, default=60, help="Refracted angle in second medium (degrees). Default: 60")
    parser.add_argument("--DT0", type=safe_float, default=10, help="Height of array center above interface (mm). Default: 10")
    parser.add_argument("--DF", type=safe_float, default=10, help="Focal depth in second medium (mm). Default: 10")
    parser.add_argument("--c1", type=safe_float, default=1480, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--c2", type=safe_float, default=5900, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--plt", type=lambda s: s.lower(), choices=["y", "n"], default="y",
                        help="Plot delay curves? y/n. Default: y")

    args = parser.parse_args()

    # Compute delays using the service layer.
    delays = run_delay_laws2D_int_service(args.M, args.s, args.angt, args.ang20, args.DT0, args.DF, args.c1, args.c2, args.plt)
    
    print("Computed delays (in microseconds):")
    print(delays)
    
    # Plot the delays vs. element index using a stem plot.
    if args.plt.lower() == 'y':
        element_indices = np.arange(1, args.M + 1)
        plt.figure(figsize=(8, 5))
        markerline, stemlines, baseline = plt.stem(element_indices, delays)
        plt.setp(markerline, 'markerfacecolor', 'b')
        plt.xlabel("Element Index")
        plt.ylabel("Delay (µs)")
        if np.isinf(args.DF):
            plot_title = "Delay Laws - Steering-only case (Stem Plot)"
        else:
            plot_title = "Delay Laws - Steering and Focusing (Stem Plot)"
        plt.title(plot_title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# interface/discrete_windows_interface.py

"""
Module: discrete_windows_interface.py
Layer: Interface

Provides a CLI to compute discrete apodization amplitudes for M elements
of a chosen window type: 'cos', 'Han', 'Ham', 'Blk', 'tri', 'rect'.

Default values:
  M=16
  type='Blk'
  plot='Y'

Example usage:
  1) python interface/discrete_windows_interface.py --M 16 --type Han --plot Y
  2) python interface/discrete_windows_interface.py --M 10 --type rect --plot N
  3) python interface/discrete_windows_interface.py               (all defaults)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.discrete_windows_service import run_discrete_windows_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Generate discrete window amplitudes for M elements of a chosen window type.",
        epilog=(
            "Example usage:\n"
            "  1) python interface/discrete_windows_interface.py --M 16 --type Han --plot Y\n"
            "  2) python interface/discrete_windows_interface.py --M 10 --type rect --plot N\n"
            "  3) python interface/discrete_windows_interface.py               (all defaults)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--M", type=int, default=16,
                        help="Number of elements (>=1). Default=16.")
    parser.add_argument("--type", type=str, default="Blk",
                        help="Window type: 'cos', 'Han', 'Ham', 'Blk', 'tri', or 'rect'. Default='Blk'")
    parser.add_argument("--outfile", type=str, default="discrete_windows_output.txt",
                        help="Output file to save the amplitudes. Default=discrete_windows_output.txt")
    parser.add_argument("--plot", type=str, choices=["Y","N","y","n"], default="Y",
                        help="If 'Y', display a stem plot of the amplitudes. Default='Y'")

    args = parser.parse_args()

    try:
        amp = run_discrete_windows_service(args.M, args.type)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save results to file
    with open(args.outfile, "w") as f:
        for i, val in enumerate(amp, start=1):
            f.write(f"Element {i}: {val:.6f}\n")
    print(f"Window amplitudes saved to {args.outfile}")

    # Optional plot
    if args.plot.upper() == "Y":
        plt.figure(figsize=(10, 6))
        x_coords = np.arange(1, len(amp)+1)
        plt.stem(x_coords, amp, linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.xlabel("Element index", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.title(f"Discrete Window: {args.type} (M={args.M})", fontsize=18)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid for both major and minor ticks
        plt.minorticks_on()  # Enable minor ticks
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# interface/fresnel_2D_and_gauss_2D_pressure_interface.py

"""
Module: fresnel_2D_and_gauss_2D_pressure_interface.py
Layer: Interface

Compare normalized pressure fields computed using Fresnel 2D and Gauss 2D models
for a 1-D piston element. The Fresnel integral approach gives a continuous field,
while Gauss 2D uses a multi-Gaussian approximation.

Example usage:
  python interface/fresnel_2D_and_gauss_2D_pressure_interface.py --b 6 --f 5 --c 1500 --z 60 --x1="-10,10,200" --x2="-10,10,40"

Defaults:
  b = 6 mm, f = 5 MHz, c = 1500 m/s, z = 60 mm,
  x1 = linspace(-10,10,200) for Fresnel 2D,
  x2 = linspace(-10,10,40) for Gauss 2D
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.fresnel_2D_service import run_fresnel_2D_service
from application.gauss_2D_service import run_gauss_2D_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compare Fresnel 2D and Gauss 2D pressure fields for a 1-D piston element.",
        epilog=(
            "Example usage:\n"
            "  python interface/fresnel_2D_and_gauss_2D_pressure_interface.py --b 6 --f 5 --c 1500 --z 60 --x1=\"-10,10,200\" --x2=\"-10,10,40\"\n\n"
            "Defaults:\n"
            "  b = 6 mm, f = 5 MHz, c = 1500 m/s, z = 60 mm,\n"
            "  x1 = linspace(-10,10,200) for Fresnel 2D,\n"
            "  x2 = linspace(-10,10,40) for Gauss 2D"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--b", type=safe_float, default=6.0, help="Half-length of the transducer (mm). Default: 6.0")
    parser.add_argument("--f", type=safe_float, default=5.0, help="Frequency (MHz). Default: 5.0")
    parser.add_argument("--c", type=safe_float, default=1500.0, help="Speed of sound (m/s). Default: 1500.0")
    parser.add_argument("--z", type=safe_float, default=60.0, help="Fixed depth (mm). Default: 60.0")
    parser.add_argument("--x1", type=str, default=None, help="x-range for Fresnel 2D (e.g., \"-10,10,200\")")
    parser.add_argument("--x2", type=str, default=None, help="x-range for Gauss 2D (e.g., \"-10,10,40\")")
    parser.add_argument("--plotfile", type=str, default=None, help="Optional file to save the plot.")

    args = parser.parse_args()

    try:
        x1_vals = parse_array(args.x1) if args.x1 else np.linspace(-10, 10, 200)
        x2_vals = parse_array(args.x2) if args.x2 else np.linspace(-10, 10, 40)
    except Exception as e:
        parser.error(str(e))

    # Compute pressure fields
    p_fresnel = run_fresnel_2D_service(args.b, args.f, args.c, x1_vals, args.z)
    p_gauss = run_gauss_2D_service(args.b, args.f, args.c, x2_vals, args.z)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x1_vals, np.abs(p_fresnel), label="Fresnel 2D", color="blue", lw=2)
    plt.plot(x2_vals, np.abs(p_gauss), 'o', label="Gauss 2D", color="red", markersize=4)
    plt.xlabel("x (mm)", fontsize=16)
    plt.ylabel("Normalized pressure magnitude", fontsize=16)
    plt.title(f"Lateral pressure field comparison at z = {args.z} mm", fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tick_params(axis='both', labelsize=14)    
    plt.legend(fontsize=14)

    if args.plotfile:
        plt.savefig(args.plotfile)
        print(f"Plot saved to {args.plotfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

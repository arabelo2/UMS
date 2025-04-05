#!/usr/bin/env python
# interface/mls_array_model_int_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from interface.cli_utils import safe_float, parse_array
from application.mls_array_model_int_service import run_mls_array_model_int_service

def main():
    parser = argparse.ArgumentParser(
        description="Simulate the MLS Array Modeling process at a fluid/fluid interface using ls_2Dint.",
        epilog=(
            "Example usage:\n"
            "  python interface/mls_array_model_int_interface.py --f 5 --d1 1.0 --c1 1480 --d2 7.9 --c2 5900 "
            "--M 32 --d 0.25 --g 0.05 --angt 0 --ang20 30 --DF 8 --DT0 25.4 --wtype rect --plot y "
            "--x=\"-5,15,200\" --z=\"1,20,200\"\n\n"
            "If --x and --z are not provided (or provided as an empty string), the domain defaults "
            "(linspace(-5,15,200) for x and linspace(1,20,200) for z) are used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--f", type=safe_float, default=5, help="Frequency in MHz. Default: 5")
    parser.add_argument("--d1", type=safe_float, default=1.0, help="Density of first medium (gm/cm^3). Default: 1.0")
    parser.add_argument("--c1", type=safe_float, default=1480, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--d2", type=safe_float, default=7.9, help="Density of second medium (gm/cm^3). Default: 7.9")
    parser.add_argument("--c2", type=safe_float, default=5900, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--M", type=int, default=32, help="Number of elements. Default: 32")
    parser.add_argument("--d", type=safe_float, default=0.25, help="Element length (mm). Default: 0.25")
    parser.add_argument("--g", type=safe_float, default=0.05, help="Gap length (mm). Default: 0.05")
    parser.add_argument("--angt", type=safe_float, default=0, help="Array angle (degrees). Default: 0")
    parser.add_argument("--ang20", type=safe_float, default=30, help="Steering angle in second medium (degrees). Default: 30")
    parser.add_argument("--DF", type=safe_float, default=8, help="Focal depth in second medium (mm). Default: 8")
    parser.add_argument("--DT0", type=safe_float, default=25.4, help="Distance from array to interface (mm). Default: 25.4")
    parser.add_argument("--wtype", type=str, default="rect", choices=["cos", "Han", "Ham", "Blk", "tri", "rect"],
                        help="Amplitude weighting function type. Default: rect")
    parser.add_argument("--plot", type=lambda s: s.lower(), choices=["y", "n"], default="y",
                        help="Plot the pressure field? y/n. Default: y")
    parser.add_argument("--x", type=str, nargs='?', default=None,
                        help=("Optional x-coordinates for field calculations as a comma-separated list. "
                              "If exactly three numbers are provided, they are interpreted as start, stop, and number of points. "
                              "Pass the value using an equal sign with no space (e.g., --x=\"-5,15,200\") to avoid misinterpretation. "
                              "Default: use domain defaults (linspace(-5,15,200))."))
    parser.add_argument("--z", type=str, nargs='?', default=None,
                        help=("Optional z-coordinates for field calculations as a comma-separated list. "
                              "If exactly three numbers are provided, they are interpreted as start, stop, and number of points. "
                              "Pass the value using an equal sign with no space (e.g., --z=\"1,20,200\") to avoid misinterpretation. "
                              "Default: use domain defaults (linspace(1,20,200))."))

    args = parser.parse_args()

    # Process x and z
    try:
        x_vals = parse_array(args.x) if args.x and args.x.strip() else None
        z_vals = parse_array(args.z) if args.z and args.z.strip() else None
    except Exception as e:
        parser.error(str(e))

    result = run_mls_array_model_int_service(
        args.f, args.d1, args.c1, args.d2, args.c2,
        args.M, args.d, args.g, args.angt, args.ang20,
        args.DF, args.DT0, args.wtype, x_vals, z_vals
    )
    p, x, z = result['p'], result['x'], result['z']

    outfile = "mls_array_model_int_output.txt"
    with open(outfile, "w") as f:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                f.write(f"{p[i, j].real:.6f}+{p[i, j].imag:.6f}j\n")
    print(f"Results saved to {outfile}")

    if args.plot == "y":
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(p), cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
        plt.xlabel("x (mm)", fontsize=16)
        plt.ylabel("z (mm)", fontsize=16)

        # Determine medium type
        medium1 = "Fluid" if args.d1 < 2.0 else "Solid"
        medium2 = "Fluid" if args.d2 < 2.0 else "Solid"

        # Dynamic plot title with all key parameters
        if np.isinf(args.DF):
            title = (
                f"MLS Steered Beam at {medium1}/{medium2} Interface\n"
                f"(f={args.f} MHz, d1={args.d1} g/cm³, c1={args.c1} m/s, d2={args.d2} g/cm³, c2={args.c2} m/s,\n"
                f"M={args.M}, d={args.d} mm, g={args.g} mm, θ={args.angt}°, Φ={args.ang20}°, "
                f"F=∞, DT0={args.DT0} mm, Window={args.wtype})"
            )
        else:
            title = (
                f"MLS Steered + Focused Beam at {medium1}/{medium2} Interface\n"
                f"(f={args.f} MHz, d1={args.d1} g/cm³, c1={args.c1} m/s, d2={args.d2} g/cm³, c2={args.c2} m/s,\n"
                f"M={args.M}, d={args.d} mm, g={args.g} mm, θ={args.angt}°, Φ={args.ang20}°, "
                f"F={args.DF} mm, DT0={args.DT0} mm, Window={args.wtype})"
            )

        plt.title(title, fontsize=16, linespacing=1.2)
        cbar = plt.colorbar()
        cbar.set_label("Normalized pressure magnitude", fontsize=16, linespacing=1.2)
        cbar.ax.tick_params(labelsize=14)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

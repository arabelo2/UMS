#!/usr/bin/env python3
# interface/ps_3Dint_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Module: ps_3Dint_interface.py
Layer: Interface

Provides a command-line interface (CLI) for computing the velocity components (vx, vy, vz)
using the ps_3Dint algorithm. This algorithm computes the normalized velocity components
for a rectangular array element radiating waves through a fluid-solid interface.

Default values:
    lx   = 6 (mm)
    ly   = 12 (mm)
    f    = 5 (MHz)
    mat  = [1, 1480, 7.9, 5900, 3200, 'p']  (material properties: d1, cp1, d2, cp2, cs2, wave_type)
    angt = 10.217 (degrees)
    Dt0  = 50.8 (mm)
    x    = linspace(0, 30, 100) (mm)
    z    = linspace(1, 20, 100) (mm)
    y    = 0 (fixed value)

Example usage:
    python interface/ps_3Dint_interface.py --lx 6 --ly 12 --f 5 --mat "1,1480,7.9,5900,3200,p" \
       --angt 10.217 --Dt0 50.8 --x="0,30,100" --z="1,20,100" --outfile "velocity_output.txt" --plotfile "plot.png"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.ps_3Dint_service import run_ps_3Dint_service
from interface.cli_utils import safe_eval, safe_float, parse_array

def apply_plot_style(ax=None, title=None, xlabel=None, ylabel=None, colorbar_obj=None):
    """
    Applies consistent font sizes and styles to matplotlib plots.

    Parameters:
        ax            : matplotlib Axes object (if None, current axes will be used)
        title         : str, plot title
        xlabel        : str, x-axis label
        ylabel        : str, y-axis label
        colorbar_obj  : Colorbar object, if provided its label and tick label sizes will be set.
    """
    if ax is None:
        ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if colorbar_obj:
        colorbar_obj.set_label("Normalized velocity magnitude", fontsize=16)
        colorbar_obj.ax.tick_params(labelsize=14)

def main():
    parser = argparse.ArgumentParser(
        description="Compute velocity components (vx, vy, vz) using the ps_3Dint algorithm.",
        epilog=(
            "Example usage:\n"
            "  python interface/ps_3Dint_interface.py --lx 6 --ly 12 --f 5 --mat \"1,1480,7.9,5900,3200,p\" \\\n"
            "       --angt 10.217 --Dt0 50.8 --x=\"0,30,100\" --z=\"1,20,100\" --outfile \"velocity_output.txt\" --plotfile \"plot.png\"\n\n"
            "Defaults:\n"
            "  lx   = 6 (mm)\n"
            "  ly   = 12 (mm)\n"
            "  f    = 5 (MHz)\n"
            "  mat  = [1, 1480, 7.9, 5900, 3200, 'p']\n"
            "  angt = 10.217 (degrees)\n"
            "  Dt0  = 50.8 (mm)\n"
            "  x    = linspace(0, 30, 100) (mm)\n"
            "  z    = linspace(1, 20, 100) (mm)\n"
            "  y    = 0 (fixed value)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Define default values
    parser.add_argument("--lx", type=safe_float, default=6.0)
    parser.add_argument("--ly", type=safe_float, default=12.0)
    parser.add_argument("--f", type=safe_float, default=5.0)
    parser.add_argument("--mat", type=str, default="1,1480,7.9,5900,3200,p")
    parser.add_argument("--angt", type=safe_float, default=10.217)
    parser.add_argument("--Dt0", type=safe_float, default=50.8)
    parser.add_argument("--x", type=str, default="0,30,100")
    parser.add_argument("--z", type=str, default="1,20,100")
    parser.add_argument("--y", type=safe_float, default=0.0)
    parser.add_argument("--outfile", type=str, default="velocity_output.txt")
    parser.add_argument("--plotfile", type=str, default=None)

    args = parser.parse_args()

    try:
        mat_values = [float(item.strip()) if item.strip().replace('.', '', 1).isdigit() else item.strip() for item in args.mat.split(",")]
        if len(mat_values) != 6:
            raise ValueError
    except ValueError:
        parser.error("mat must contain exactly six comma-separated values: d1,cp1,d2,cp2,cs2,wave_type (e.g., '1,1480,7.9,5900,3200,p').")

    try:
        x = parse_array(args.x)
        z = parse_array(args.z)
    except Exception as e:
        parser.error(str(e))

    xx, zz = np.meshgrid(x, z)

    vx, vy, vz = run_ps_3Dint_service(args.lx, args.ly, args.f, mat_values, 0, 0, args.angt, args.Dt0, xx, args.y, zz)
    v = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)

    # Dynamic title generation
    medium1 = "Fluid" if mat_values[0] < 2.0 else "Solid"
    medium2 = "Fluid" if mat_values[2] < 2.0 else "Solid"
    title = (
        f"Velocity Field at {medium1}/{medium2} Interface\n"
        f"f = {args.f:.2f} MHz, lx = {args.lx:.1f} mm, ly = {args.ly:.1f} mm,\n"
        f"d1 = {mat_values[0]:.2f} g/cm³, cp1 = {mat_values[1]:.0f} m/s, "
        f"d2 = {mat_values[2]:.2f} g/cm³, cp2 = {mat_values[3]:.0f} m/s, "
        f"cs2 = {mat_values[4]:.0f} m/s,\n"
        f"θ = {args.angt:.1f}°, Dt0 = {args.Dt0:.1f} mm, wave = {mat_values[5]}"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(v, cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
    cb = fig.colorbar(im, ax=ax)
    apply_plot_style(ax, title=title, xlabel="x (mm)", ylabel="z (mm)", colorbar_obj=cb)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()

    if args.plotfile:
        plt.savefig(args.plotfile, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.plotfile}")

    plt.show()

    try:
        with open(args.outfile, "w") as f:
            for row in v:
                formatted_row = "\t".join("%0.16f" % val for val in row)
                f.write(formatted_row + "\n")
        print(f"Velocity magnitude matrix saved in {args.outfile}")
    except Exception as e:
        print("Error saving the output file:", e)

if __name__ == "__main__":
    main()

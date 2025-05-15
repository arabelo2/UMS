#!/usr/bin/env python3
"""
Module: mps_array_model_int_interface.py
Layer: Interface

Provides a CLI to compute the normalized velocity field for an array of 1-D elements 
radiating waves through a fluid/solid interface using the ps_3Dint algorithm.
It utilizes the mps_array_model_int service to compute the field, then saves and 
optionally plots the results.

Default values:
  lx        = 0.15 mm
  ly        = 0.15 mm
  gx        = 0.05 mm
  gy        = 0.05 mm
  f         = 5 MHz
  d1        = 1.0
  c1        = 1480 m/s
  d2        = 7.9
  c2        = 5900 m/s
  cs2       = 3200 m/s
  type      = 'p'
  L1        = 11
  L2        = 11
  angt      = 10.217 deg
  Dt0       = 50.8 mm
  theta20   = 20 deg
  phi       = 0 deg
  DF        = inf
  ampx_type = 'rect'
  ampy_type = 'rect'
  xs        = linspace(-5,20,100)
  zs        = linspace(1,20,100)
  y         = 0 (scalar) or a vector (e.g., linspace(-5,5,100))

Example usage:
  1) With plot (2D evaluation using scalar y):
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 \
--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 \
--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y=0 --plot=y

  2) With plot (3D evaluation using vector y):
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 \
--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 \
--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y="-5,5,100" --plot=y

  3) Without plot (using scalar y):
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 \
--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 \
--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y=0 --plot=n

  4) With both 2D and additional 3D visualization (for 2D simulation):
     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 \
--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 \
--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs="-5,20,100" --zs="1,20,100" --y=0 --plot=y --plot-3dfield
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.mps_array_model_int_service import run_mps_array_model_int_service
from interface.cli_utils import safe_float, parse_array

def apply_plot_style(ax=None, title=None, xlabel=None, ylabel=None, zlabel=None, colorbar_obj=None):
    if ax is None:
        ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    if zlabel and hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(zlabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    if colorbar_obj:
        colorbar_obj.set_label("Normalized velocity magnitude", fontsize=16)
        colorbar_obj.ax.tick_params(labelsize=14)

def main():
    parser = argparse.ArgumentParser(
        description="Compute the normalized velocity field for an array at a fluid/solid interface.",
        epilog=(
            "Example usage:\n\n"
            "  1) With plot, using scalar y (2D evaluation in x-z plane):\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 "
            "--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 "
            "--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" "
            "--y=0 --plot=y\n\n"
            "  2) With plot, using vector y (3D evaluation):\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 "
            "--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 "
            "--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" "
            "--y=\"-5,5,100\" --plot=y\n\n"
            "  3) Without plot (using scalar y):\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 "
            "--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 "
            "--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" "
            "--y=0 --plot=n\n\n"
            "  4) With both 2D and additional 3D visualization (for 2D simulation):\n"
            "     python interface/mps_array_model_int_interface.py --lx=0.15 --ly=0.15 --gx=0.05 --gy=0.05 --f=5 "
            "--d1=1.0 --c1=1480 --d2=7.9 --c2=5900 --cs2=3200 --type=p --L1=11 --L2=11 --angt=10.217 --Dt0=50.8 "
            "--theta20=20 --phi=0 --DF=inf --ampx_type=rect --ampy_type=rect --xs=\"-5,20,100\" --zs=\"1,20,100\" "
            "--y=0 --plot=y --plot-3dfield\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--lx", type=safe_float, default=0.15, help="Element length in x-direction (mm). Default: 0.15")
    parser.add_argument("--ly", type=safe_float, default=0.15, help="Element length in y-direction (mm). Default: 0.15")
    parser.add_argument("--gx", type=safe_float, default=0.05, help="Gap length in x-direction (mm). Default: 0.05")
    parser.add_argument("--gy", type=safe_float, default=0.05, help="Gap length in y-direction (mm). Default: 0.05")
    parser.add_argument("--f", type=safe_float, default=5.0, help="Frequency (MHz). Default: 5")
    parser.add_argument("--d1", type=safe_float, default=1.0, help="Density of medium one. Default: 1.0")
    parser.add_argument("--c1", type=safe_float, default=1480.0, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--d2", type=safe_float, default=7.9, help="Density of medium two. Default: 7.9")
    parser.add_argument("--c2", type=safe_float, default=5900.0, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--cs2", type=safe_float, default=3200.0, help="Shear wave speed in second medium (m/s). Default: 3200")
    parser.add_argument("--type", type=str, default="p", choices=["p", "s"], help="Wave type for medium two ('p' or 's'). Default: p")
    parser.add_argument("--L1", type=int, default=11, help="Number of elements in x-direction. Default: 11")
    parser.add_argument("--L2", type=int, default=11, help="Number of elements in y-direction. Default: 11")
    parser.add_argument("--angt", type=safe_float, default=10.217, help="Array angle with interface (deg). Default: 10.217")
    parser.add_argument("--Dt0", type=safe_float, default=50.8, help="Height of array center above interface (mm). Default: 50.8")
    parser.add_argument("--theta20", type=safe_float, default=20.0, help="Steering angle in theta direction (deg). Default: 20")
    parser.add_argument("--phi", type=safe_float, default=0.0, help="Steering angle in phi direction (deg). Default: 0")
    parser.add_argument("--DF", type=safe_float, default=float('inf'), help="Focal distance (mm). Use inf for steering-only. Default: inf")
    parser.add_argument("--ampx_type", type=str, default="rect", choices=["rect", "cos", "Han", "Ham", "Blk", "tri"], help="Window type in x-direction. Default: rect")
    parser.add_argument("--ampy_type", type=str, default="rect", choices=["rect", "cos", "Han", "Ham", "Blk", "tri"], help="Window type in y-direction. Default: rect")
    parser.add_argument("--xs", type=str, default="-5,20,100", help="x-coordinates as comma-separated list or start,stop,num_points. Default: \"-5,20,100\"")
    parser.add_argument("--zs", type=str, default="1,20,100", help="z-coordinates as comma-separated list or start,stop,num_points. Default: \"1,20,100\"")
    parser.add_argument("--y", type=str, default="0", help="y-coordinate(s) for evaluation. For scalar use e.g. '0', for vector use e.g. '-5,5,100'.")
    parser.add_argument("--plot", type=lambda s: s.lower(), choices=["y", "n"], default="y", help="Plot the velocity field? (y/n). Default: y")
    # Optional flag to plot a 3D field even for 2D simulation (like ps_3Dv_interface)
    parser.add_argument("--plot-3dfield", action="store_true", help="Plot a 3D field visualization for 2D simulations.")
    parser.add_argument("--z_scale", type=safe_float, default=10.0, help="Scale factor for z-axis (delay values) in stem plot. Default: 10")
    parser.add_argument("--elev", type=safe_float, default=25.0, help="Camera elevation for 3D plot. Default: 25")
    parser.add_argument("--azim", type=safe_float, default=20.0, help="Camera azimuth for 3D plot. Default: 20")

    args = parser.parse_args()

    # Parse evaluation coordinates.
    try:
        x_vals = parse_array(args.xs)
    except Exception as e:
        parser.error(str(e))
    try:
        z_vals = parse_array(args.zs)
    except Exception as e:
        parser.error(str(e))
    try:
        y_vals = parse_array(args.y)
    except Exception as e:
        parser.error(str(e))
    
    # Helper: determine if input is a vector.
    def is_vector(val):
        arr = np.atleast_1d(val)
        return arr.size > 1

    x_is_vec = is_vector(x_vals)
    y_is_vec = is_vector(y_vals)
    z_is_vec = is_vector(z_vals)
    vec_count = sum([x_is_vec, y_is_vec, z_is_vec])
    
    # Determine simulation mode:
    # - 1D: ≤1 vector.
    # - 2D: exactly 2 vectors.
    # - 3D: all 3 are vectors.
    if vec_count <= 1:
        mode = "1D"
    elif vec_count == 2:
        mode = "2D"
    elif vec_count == 3:
        mode = "3D"
    else:
        parser.error("Unexpected coordinate combination.")
    
    # Build evaluation grid based on the mode.
    if mode == "1D":
        X_input = x_vals
        Y_input = y_vals
        Z_input = z_vals
    elif mode == "2D":
        if not x_is_vec:
            Y_grid, Z_grid = np.meshgrid(np.atleast_1d(y_vals), np.atleast_1d(z_vals))
            X_grid = np.full(Y_grid.shape, x_vals)
            X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
        elif not y_is_vec:
            X_grid, Z_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(z_vals))
            Y_grid = np.full(X_grid.shape, y_vals)
            X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
        elif not z_is_vec:
            X_grid, Y_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(y_vals))
            Z_input = z_vals
            X_input, Y_input = X_grid, Y_grid
    else:  # mode == "3D"
        X_input, Y_input, Z_input = np.meshgrid(np.atleast_1d(x_vals),
                                                 np.atleast_1d(y_vals),
                                                 np.atleast_1d(z_vals),
                                                 indexing='ij')
    
    # Compute velocity field.
    result = run_mps_array_model_int_service(
         args.lx, args.ly, args.gx, args.gy,
         args.f, args.d1, args.c1, args.d2, args.c2, args.cs2, args.type,
         args.L1, args.L2, args.angt, args.Dt0,
         args.theta20, args.phi, args.DF,
         args.ampx_type, args.ampy_type,
         x_vals, z_vals, y_vals   # Note: order as defined in the service.
    )
    
    if not isinstance(result, dict) or 'p' not in result or 'x' not in result or 'z' not in result:
        print("Error: Service did not return expected result dictionary with keys 'p', 'x', 'z'.")
        sys.exit(1)
    
    p = result['p']
    x = result['x']
    z = result['z']
    
    outfile = "mps_array_model_int_output.txt"
    with open(outfile, "w") as f:
        for idx, val in np.ndenumerate(p):
            # Ensure scalar value.
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    val = val.item()
            f.write(f"{val.real:.6f}+{val.imag:.6f}j\n")
    print(f"Results saved to {outfile}")
    
    # Build detailed plot title with medium information.
    medium1 = "Fluid" if args.d1 < 2.0 else "Solid"
    medium2 = "Fluid" if args.d2 < 2.0 else "Solid"
    title_info = f"Interface: {medium1}/{medium2} | d1={args.d1}, c1={args.c1} m/s | d2={args.d2}, c2={args.c2} m/s"
    
    Nx, Ny = args.L1, args.L2
    elem_size_ratio = args.lx/((args.c1/1e3)/args.f)    # element length in λ-units --> λ_mm = (c1_mm_per_us / args.f) --> e.g. 1.480 mm/μs ÷ 5 MHz = 0.296 mm
    wave = "P" if args.type=="p" else "S"
    title_info = (
        f"Normalized velocity magnitude in steel for an "
        f"{Nx}×{Ny} 2-D phased-array\n"
        f"(element size ≈ {elem_size_ratio:.2f} λ) "
        f"normal to the interface at Dt₀ = {args.Dt0} mm,\n"
        f"radiating a {wave}-wave through a water–steel interface."
    )

    # Plotting.
    if mode == "1D":
        plt.figure(figsize=(8,5))
        if x_is_vec:
            independent = x_vals
            xlabel = "x (mm)"
        elif y_is_vec:
            independent = y_vals
            xlabel = "y (mm)"
        else:
            independent = z_vals
            xlabel = "z (mm)"
        plt.plot(independent, np.abs(p), 'b-', lw=2)
        ax = plt.gca()
        apply_plot_style(ax, title=f"1D Simulation\n{title_info}", xlabel=xlabel, ylabel="Normalized velocity magnitude")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
    elif mode == "2D":
        # If both plot and plot-3dfield options are provided, produce both.
        if args.plot == "y":
            plt.figure(figsize=(8,6))
            if not x_is_vec:
                independent1 = np.atleast_1d(y_vals)
                independent2 = np.atleast_1d(z_vals)
                extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
                im = plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
                plt.xlabel("y (mm)")
                plt.ylabel("z (mm)")
                plt.title(f"2D Simulation at x = {x_vals} mm\n{title_info}")
                ax = plt.gca()
                apply_plot_style(ax, title=f"2D Simulation at x = {x_vals} mm\n{title_info}", xlabel="y (mm)", ylabel="z (mm)")
            elif not y_is_vec:
                independent1 = np.atleast_1d(x_vals)
                independent2 = np.atleast_1d(z_vals)
                extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
                im = plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
                plt.xlabel("x (mm)")
                plt.ylabel("z (mm)")
                plt.title(f"2D Simulation at y = {y_vals} mm\n{title_info}")
                ax = plt.gca()
                apply_plot_style(ax, title=f"2D Simulation at y = {y_vals} mm\n{title_info}", xlabel="x (mm)", ylabel="z (mm)")
            elif not z_is_vec:
                independent1 = np.atleast_1d(x_vals)
                independent2 = np.atleast_1d(y_vals)
                extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
                im = plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                plt.title(f"2D Simulation at z = {z_vals} mm\n{title_info}")
                ax = plt.gca()
                apply_plot_style(ax, title=f"2D Simulation at z = {z_vals} mm\n{title_info}", xlabel="x (mm)", ylabel="y (mm)")
            cb = plt.colorbar(label="Normalized velocity magnitude")
            cb.ax.tick_params(labelsize=14)
            cb.set_label("Normalized velocity magnitude", fontsize=16)
        if args.plot_3dfield:
            # Additional 3D visualization for 2D simulation.
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if not x_is_vec:
                Y_grid, Z_grid = np.meshgrid(np.atleast_1d(y_vals), np.atleast_1d(z_vals))
                sc = ax.scatter(Y_grid.flatten(), Z_grid.flatten(), np.abs(p.flatten()), c=np.abs(p.flatten()), cmap="jet", s=5)
                ax.set_xlabel("y (mm)")
                ax.set_ylabel("z (mm)")
                ax.set_zlabel("Velocity")
                ax.set_title(f"3D Visualization at x = {x_vals}\n{title_info}")
                apply_plot_style(ax, title=f"3D Visualization at x = {x_vals}\n{title_info}", xlabel="y (mm)", ylabel="z (mm)", zlabel="Velocity")
            elif not y_is_vec:
                X_grid, Z_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(z_vals))
                sc = ax.scatter(X_grid.flatten(), Z_grid.flatten(), np.abs(p.flatten()), c=np.abs(p.flatten()), cmap="jet", s=5)
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("z (mm)")
                ax.set_zlabel("Velocity")
                ax.set_title(f"3D Visualization at y = {y_vals}\n{title_info}")
                apply_plot_style(ax, title=f"3D Visualization at y = {y_vals}\n{title_info}", xlabel="x (mm)", ylabel="z (mm)", zlabel="Velocity")
            elif not z_is_vec:
                X_grid, Y_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(y_vals))
                sc = ax.scatter(X_grid.flatten(), Y_grid.flatten(), np.abs(p.flatten()), c=np.abs(p.flatten()), cmap="jet", s=5)
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
                ax.set_zlabel("Velocity")
                ax.set_title(f"3D Visualization at z = {z_vals}\n{title_info}")
                apply_plot_style(ax, title=f"3D Visualization at z = {z_vals}\n{title_info}", xlabel="x (mm)", ylabel="y (mm)", zlabel="Velocity")
            cb = fig.colorbar(ax.collections[0], ax=ax, label="Normalized velocity magnitude")
            apply_plot_style(ax, colorbar_obj=cb)
    else:  # mode == "3D"
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_input.flatten(), Y_input.flatten(), Z_input.flatten(), 
                   c=np.abs(p.flatten()), cmap="jet", marker='o', s=5)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_title(f"3D Simulation for Array Velocity Field\n{title_info}")
        apply_plot_style(ax, title=f"3D Simulation for Array Velocity Field\n{title_info}", xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
        cb = fig.colorbar(ax.collections[0], ax=ax, label="Normalized velocity magnitude")
        apply_plot_style(ax, colorbar_obj=cb)

    plt.show()

if __name__ == "__main__":
    main()

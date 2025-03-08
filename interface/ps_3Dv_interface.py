#!/usr/bin/env python
# interface/ps_3Dv_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from application.ps_3Dv_service import run_ps_3Dv_service
from interface.cli_utils import safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="Compute normalized pressure for a rectangular piston using the 3D Rayleigh-Sommerfeld integral.",
        epilog=(
            "Example usage:\n\n"
            "1. 1D simulation (x and y are scalars, z is a vector):\n"
            "   python interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 \\\n"
            "       --x=0 --y=0 --z=\"5,100,400\"\n\n"
            "2. 2D simulation with (vector, vector, scalar):\n"
            "   python interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 \\\n"
            "       --x=\"-5,5,200\" --y=\"-5,5,200\" --z=50\n\n"
            "3. 2D simulation with (vector, scalar, vector):\n"
            "   python interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 \\\n"
            "       --x=\"-5,5,200\" --y=0 --z=\"5,100,400\"\n\n"
            "4. 2D simulation with (scalar, vector, vector):\n"
            "   python interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 \\\n"
            "       --x=0 --y=\"-5,5,200\" --z=\"5,100,400\"\n\n"
            "5. 3D simulation (all three coordinates as vectors):\n"
            "   python interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 \\\n"
            "       --x=\"-5,5,50\" --y=\"-5,5,50\" --z=\"5,100,50\"\n\n"
            "Note: For array inputs, if exactly three numbers are provided, they are interpreted\n"
            "      as start, stop, and number of points (via np.linspace); otherwise, the values\n"
            "      are taken as explicit entries. For negative values, enclose the argument in quotes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--lx", type=safe_float, default=6, help="Element length along x (mm). Default: 6")
    parser.add_argument("--ly", type=safe_float, default=12, help="Element length along y (mm). Default: 12")
    parser.add_argument("--f", type=safe_float, default=5, help="Frequency (MHz). Default: 5")
    parser.add_argument("--c", type=safe_float, default=1480, help="Wave speed (m/s). Default: 1480")
    parser.add_argument("--ex", type=safe_float, default=0, help="Offset along x (mm). Default: 0")
    parser.add_argument("--ey", type=safe_float, default=0, help="Offset along y (mm). Default: 0")
    parser.add_argument("--x", type=str, default="0", help="x-coordinate(s) for evaluation. For 1D, scalar; for 2D, comma-separated list.")
    parser.add_argument("--y", type=str, default="0", help="y-coordinate(s) for evaluation. For 1D, scalar; for 2D, comma-separated list.")
    parser.add_argument("--z", type=str, default="5,100,400", help="z-coordinate(s) for evaluation. If three numbers, interpreted as linspace; else explicit values.")
    parser.add_argument("--P", type=int, default=None, help="Optional number of segments in x-direction.")
    parser.add_argument("--Q", type=int, default=None, help="Optional number of segments in y-direction.")

    args = parser.parse_args()

    # Parse evaluation coordinates.
    try:
        if "," in args.x:
            x_vals = parse_array(args.x)
        else:
            x_vals = safe_float(args.x)
    except Exception as e:
        parser.error(f"Invalid x coordinate: {args.x}. Error: {str(e)}")
    
    try:
        if "," in args.y:
            y_vals = parse_array(args.y)
        else:
            y_vals = safe_float(args.y)
    except Exception as e:
        parser.error(f"Invalid y coordinate: {args.y}. Error: {str(e)}")
    
    try:
        if "," in args.z:
            z_vals = parse_array(args.z)
        else:
            z_vals = safe_float(args.z)
    except Exception as e:
        parser.error(f"Invalid z coordinate: {args.z}. Error: {str(e)}")
    
    # Helper: determine if input is a vector (more than one element)
    def is_vector(val):
        arr = np.atleast_1d(val)
        return arr.size > 1

    # Determine which coordinates are vectors.
    x_is_vec = is_vector(x_vals)
    y_is_vec = is_vector(y_vals)
    z_is_vec = is_vector(z_vals)
    vec_count = sum([x_is_vec, y_is_vec, z_is_vec])
    
    # Determine simulation mode:
    # - 1D simulation: one coordinate is vector (others scalars)
    # - 2D simulation: exactly two are vectors
    # - 3D simulation: all three are vectors
    if vec_count <= 1:
        mode = "1D"
    elif vec_count == 2:
        mode = "2D"
    elif vec_count == 3:
        mode = "3D"
    else:
        parser.error("Unexpected coordinate combination.")
    
    # Prepare inputs for the service function.
    if mode == "1D":
        X_input = x_vals
        Y_input = y_vals
        Z_input = z_vals
    elif mode == "2D":
        # Two coordinates are vectors: form a meshgrid from the two vectors
        if not x_is_vec:
            # x is scalar; y and z are vectors.
            Y_grid, Z_grid = np.meshgrid(np.atleast_1d(y_vals), np.atleast_1d(z_vals))
            X_grid = np.full(Y_grid.shape, x_vals)
            X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
        elif not y_is_vec:
            # y is scalar; x and z are vectors.
            X_grid, Z_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(z_vals))
            Y_grid = np.full(X_grid.shape, y_vals)
            X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
        elif not z_is_vec:
            # z is scalar; x and y are vectors.
            X_grid, Y_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(y_vals))
            Z_input = z_vals  # scalar
            X_input, Y_input = X_grid, Y_grid
    else:  # mode == "3D"
        # All three are vectors: create a full 3D meshgrid.
        X_input, Y_input, Z_input = np.meshgrid(np.atleast_1d(x_vals),
                                                 np.atleast_1d(y_vals),
                                                 np.atleast_1d(z_vals),
                                                 indexing='ij')
    
    # Compute pressure.
    p = run_ps_3Dv_service(args.lx, args.ly, args.f, args.c, args.ex, args.ey, X_input, Y_input, Z_input, args.P, args.Q)
    
    # Plotting.
    if mode == "1D":
        plt.figure(figsize=(8,5))
        # Determine independent axis.
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
        plt.xlabel(xlabel)
        plt.ylabel("Normalized Pressure Magnitude")
        plt.title("1D Rayleigh-Sommerfeld Simulation for Rectangular Piston")
        plt.grid(True)
    elif mode == "2D":
        # For 2D simulation, use the two independent vectors to set the extent.
        if not x_is_vec:
            independent1 = np.atleast_1d(y_vals)
            independent2 = np.atleast_1d(z_vals)
            extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
            plt.figure(figsize=(8,6))
            plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
            plt.xlabel("y (mm)")
            plt.ylabel("z (mm)")
            plt.title(f"2D Simulation at x = {x_vals}")
        elif not y_is_vec:
            independent1 = np.atleast_1d(x_vals)
            independent2 = np.atleast_1d(z_vals)
            extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
            plt.figure(figsize=(8,6))
            plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
            plt.xlabel("x (mm)")
            plt.ylabel("z (mm)")
            plt.title(f"2D Simulation at y = {y_vals}")
        elif not z_is_vec:
            independent1 = np.atleast_1d(x_vals)
            independent2 = np.atleast_1d(y_vals)
            extent = [independent1.min(), independent1.max(), independent2.max(), independent2.min()]
            plt.figure(figsize=(8,6))
            plt.imshow(np.abs(p), cmap="jet", extent=extent, aspect='auto')
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.title(f"2D Simulation at z = {z_vals}")
        plt.colorbar(label="Pressure Magnitude")
    else:  # mode == "3D"
        # For full 3D, use a 3D scatter plot (similar to MATLAB's plot3).
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_input.flatten(), Y_input.flatten(), Z_input.flatten(), 
                        c=np.abs(p.flatten()), cmap="jet", marker='o', s=5)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_title("3D Rayleigh-Sommerfeld Simulation for Rectangular Piston")
        fig.colorbar(sc, ax=ax, label="Pressure Magnitude")
    
    plt.show()

if __name__ == "__main__":
    main()

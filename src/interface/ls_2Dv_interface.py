# interface/ls_2Dv_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from application.ls_2Dv_service import run_ls_2Dv_service
from interface.cli_utils import safe_eval, safe_float, parse_array

def main():
    parser = argparse.ArgumentParser(
        description="LS 2-D Simulation Interface",
        epilog=(
            "Example usage:\n"
            "  python interface/ls_2Dv_interface.py --b 1 --f 5 --c 1500 --e 0 --x 0 --z '5,80,200' \\\n"
            "    --x2=\"-10,10,200\" --z2=\"1,25,200\" --plot-mode both --N 20\n\n"
            "By default:\n"
            "  1D simulation uses x=0 and z=linspace(5,80,200).\n"
            "  2D simulation uses x=linspace(-10,10,200) and z=linspace(1,25,200).\n"
            "The --N option specifies the number of segments to use. If not provided, it is computed automatically "
            "to yield one segment per wavelength. For example, use --N 20 to force 20 segments.\n"
            "Note: For negative values in --x2 or --z2, use the equals sign (e.g., --x2=\"-10,10,200\").\n"
            "The --plot-mode option accepts 'both' (default), '1D', or '2D'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 1D simulation parameters.
    parser.add_argument("--b", type=safe_float, default=1,
                        help="Half-length of the element (mm). Default: 1")
    parser.add_argument("--f", type=safe_float, default=5,
                        help="Frequency (MHz). Default: 5")
    parser.add_argument("--c", type=safe_float, default=1500,
                        help="Wave speed (m/s). Default: 1500")
    parser.add_argument("--e", type=safe_float, default=0,
                        help="Lateral offset of the element's center (mm). Default: 0")
    parser.add_argument("--x", type=safe_float, default=0,
                        help="x-coordinate (mm) for 1D simulation. Default: 0")
    parser.add_argument("--z", type=str, default=None,
                        help=(
                            "z-coordinate(s) (mm) for 1D simulation. "
                            "Provide a comma-separated list of numbers. If exactly three numbers are provided, they are interpreted as start, stop, and number of points (e.g., '5,80,200'). "
                            "Otherwise, they are taken as explicit values. "
                            "Default: 200 points linearly spaced between 5 and 80 mm."
                        ))
    parser.add_argument("--N", type=int, default=None,
                        help="Optional number of segments for numerical integration. If not provided, computed automatically (one segment per wavelength).")
    
    # 2D simulation parameters.
    parser.add_argument("--x2", type=str, default=None,
                        help=(
                            "x-coordinate(s) (mm) for 2D simulation. "
                            "Provide a comma-separated list of numbers. If exactly three numbers are provided, they are interpreted as start, stop, and number of points (e.g., '-10,10,200'). "
                            "Otherwise, they are taken as explicit values. "
                            "Default: 200 points linearly spaced between -10 and 10. "
                            "Note: For negative values, use the equals sign, e.g. --x2=\"-10,10,200\"."
                        ))
    parser.add_argument("--z2", type=str, default=None,
                        help=(
                            "z-coordinate(s) (mm) for 2D simulation. "
                            "Provide a comma-separated list of numbers. If exactly three numbers are provided, they are interpreted as start, stop, and number of points (e.g., '1,25,200'). "
                            "Otherwise, they are taken as explicit values. "
                            "Default: 200 points linearly spaced between 1 and 20. "
                            "Note: For negative values, use the equals sign if needed."
                        ))
    
    # Option to choose which plot(s) to show.
    parser.add_argument("--plot-mode", type=str, choices=["both", "1D", "2D"], default="both",
                        help="Plot mode: 'both' for both 1D and 2D, '1D' for 1D only, or '2D' for 2D only. Default: both.")
    
    args = parser.parse_args()
    
    # Process 1D simulation z parameter.
    try:
        if args.z is None:
            z1 = np.linspace(5, 80, 200)
        else:
            z1 = parse_array(args.z)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    
    p1 = None
    if args.plot_mode in ["both", "1D"]:
        p1 = run_ls_2Dv_service(args.b, args.f, args.c, args.e, args.x, z1, args.N)
        # Squeeze extra dimension if present.
        if isinstance(p1, np.ndarray) and p1.ndim > 0 and p1.shape[0] == 1:
            p1 = np.squeeze(p1, axis=0)
    
    p2 = None
    if args.plot_mode in ["both", "2D"]:
        if args.x2 is None:
            x2 = np.linspace(-10, 10, 200)
        else:
            try:
                x2 = parse_array(args.x2)
            except argparse.ArgumentTypeError as e:
                parser.error(str(e))
        if args.z2 is None:
            z2 = np.linspace(1, 25, 200)
        else:
            try:
                z2 = parse_array(args.z2)
            except argparse.ArgumentTypeError as e:
                parser.error(str(e))
        xx, zz = np.meshgrid(x2, z2)
        p2 = run_ls_2Dv_service(args.b, args.f, args.c, args.e, xx, zz, args.N)
        # Squeeze extra integration dimension if necessary.
        if isinstance(p2, np.ndarray) and p2.ndim > 2 and p2.shape[0] == 1:
            p2 = np.squeeze(p2, axis=0)
    
    if args.plot_mode in ["both", "1D"]:
        plt.figure(figsize=(8, 5))
        plt.plot(z1, np.abs(p1), 'b-', lw=2)
        plt.xlabel("z (mm)", fontsize=18)  # Set fontsize for x-axis label        
        plt.ylabel("Normalized pressure magnitude", fontsize=18)  # Set fontsize for y-axis label
        if args.N == 1:
            plt.title("Normalized pressure field for a single line source element", fontsize=18)
        elif args.N and args.N > 1:
            plt.title(f"Normalized pressure calculation for a 1-D piston element using the Line Source Model (N = {args.N})", fontsize=18)
        else:
            plt.title("Normalized pressure calculation for a 1-D piston element using the Line Source Model", fontsize=18)
        plt.tick_params(axis='both', labelsize=16)  # Set fontsize for tick labels on both axes
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid for both major and minor ticks
        plt.minorticks_on()  # Enable minor ticks
    
    if args.plot_mode in ["both", "2D"]:
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(p2), cmap="jet",
                   extent=[x2.min(), x2.max(), z2.max(), z2.min()],
                   aspect="auto")
        plt.xlabel("x (mm)", fontsize=16)  # Set fontsize for x-axis label
        plt.ylabel("z (mm)", fontsize=16)  # Set fontsize for y-axis label
        if args.N == 1:
            plt.title("Normalized pressure field for a 2-D single line source element", fontsize=18)
        elif args.N and args.N > 1:
            plt.title(f"Normalized pressure calculation for a 2-D piston element using the Line Source Model (N = {args.N})", fontsize=18)
        else:
            plt.title("Normalized pressure calculation for a 2-D piston element using the Line Source Model", fontsize=18)
        cbar = plt.colorbar()  # Create colorbar
        cbar.set_label("Normalized pressure magnitude", fontsize=16)  # Set fontsize for colorbar label
        cbar.ax.tick_params(labelsize=14)  # Set fontsize for colorbar tick labels
        plt.tick_params(axis='both', labelsize=16)  # Set fontsize for tick labels on both axes
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enable grid for both major and minor ticks
        plt.minorticks_on()  # Enable minor ticks
    
    plt.show()

if __name__ == '__main__':
    main()

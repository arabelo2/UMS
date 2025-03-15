#!/usr/bin/env python3
"""
Module: delay_laws3Dint_interface.py
Layer: Interface

Provides a CLI to compute delay laws for a 2D array at a fluid/solid interface in 3D.
It uses the delay_laws3Dint service to calculate the time delays and optionally plot ray geometry.

Default values:
  Mx       = 4
  My       = 4
  sx       = 0.5 mm
  sy       = 0.5 mm
  theta    = 20 deg       (array angle with interface)
  phi      = 0 deg        (steering parameter for second medium)
  theta20  = 45 deg       (refracted steering angle in second medium)
  DT0      = 10 mm
  DF       = 10 mm        (focal distance; use inf for steering-only)
  c1       = 1480 m/s
  c2       = 5900 m/s
  plt      = 'y'          (plot ray geometry)
  elev     = 25           (camera elevation for plot)
  azim     = 20           (camera azimuth for plot)
  z_scale  = 10           (scale factor for z-axis in stem plot)
  
Example usage:
  1) Steering and focusing with plot:
     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=y --elev=25 --azim=20 --z_scale=10
     
  2) Steering and focusing without plot:
     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=n
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute delay laws for a 2D array at a fluid/solid interface in 3D.",
        epilog=(
            "Example usage:\n"
            "  1) With plot:\n"
            "     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=y --elev=25 --azim=20 --z_scale=10\n\n"
            "  2) Without plot:\n"
            "     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--Mx", type=int, default=4, help="Number of elements in x-direction. Default: 4")
    parser.add_argument("--My", type=int, default=4, help="Number of elements in y-direction. Default: 4")
    parser.add_argument("--sx", type=safe_float, default=0.5, help="Pitch in x-direction (mm). Default: 0.5")
    parser.add_argument("--sy", type=safe_float, default=0.5, help="Pitch in y-direction (mm). Default: 0.5")
    parser.add_argument("--theta", type=safe_float, default=20.0, help="Array angle with interface (deg). Default: 20")
    parser.add_argument("--phi", type=safe_float, default=0.0, help="Steering angle phi (deg). Default: 0")
    parser.add_argument("--theta20", type=safe_float, default=45.0, help="Refracted steering angle in second medium (deg). Default: 45")
    parser.add_argument("--DT0", type=safe_float, default=10.0, help="Height of array center above interface (mm). Default: 10")
    parser.add_argument("--DF", type=safe_float, default=10.0, help="Focal distance in second medium (mm). Default: 10")
    parser.add_argument("--c1", type=safe_float, default=1480.0, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--c2", type=safe_float, default=5900.0, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--plt", type=lambda s: s.lower(), choices=["y", "n"], default="y", help="Plot ray geometry? (y/n). Default: y")
    parser.add_argument("--elev", type=safe_float, default=25.0, help="Camera elevation for 3D plot. Default: 25")
    parser.add_argument("--azim", type=safe_float, default=20.0, help="Camera azimuth for 3D plot. Default: 20")
    parser.add_argument("--z_scale", type=safe_float, default=10.0, help="Scale factor for z-axis (delay values) in stem plot. Default: 10")
    
    args = parser.parse_args()
    
    try:
        td = run_delay_laws3Dint_service(
            args.Mx, args.My, args.sx, args.sy,
            args.theta, args.phi, args.theta20,
            args.DT0, args.DF, args.c1, args.c2,
            args.plt
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Computed delays (in microseconds):")
    print(td)
    
    # Plot a 3D stem plot if plotting is enabled
    if args.plt.upper() == "Y":
        plot_title = "Delay Laws 3DInt - 3D Stem Plot"
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scale delays to exaggerate vertical variation
        td_scaled = td * args.z_scale
        
        # Generate grid indices for plotting
        X, Y_indices = np.meshgrid(range(args.My), range(args.Mx))
        
        # Emulate MATLAB's stem3: for each element, draw a vertical line from 0 to scaled delay value.
        for i in range(args.Mx):
            for j in range(args.My):
                ax.plot([j, j], [i, i], [0, td_scaled[i, j]], marker='o', color='b')
        
        ax.set_xlabel("Element index (y-direction)")
        ax.set_ylabel("Element index (x-direction)")
        ax.set_zlabel("Scaled Time Delay (µs)")
        ax.set_title(plot_title)
        
        # Set aspect ratio based on data ranges
        z_range = np.ptp(td_scaled)
        ax.set_box_aspect([args.My, args.Mx, z_range if z_range > 0 else 1])
        
        # Set view using CLI parameters
        ax.view_init(elev=args.elev, azim=args.azim)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

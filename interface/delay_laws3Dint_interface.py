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
  theta    = 20 deg
  phi      = 0 deg
  theta20  = 45 deg
  DT0      = 10 mm
  DF       = 10 mm
  c1       = 1480 m/s
  c2       = 5900 m/s
  plt      = 'y' (plot rays)
  
Example usage:
  1) Steering and focusing with plot:
     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=y
     
  2) Steering and focusing without plot:
     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=n

Note: The view angle for plotting is set to elev=25 and azim=20 by default.
"""

import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from interface.cli_utils import safe_float

def main():
    parser = argparse.ArgumentParser(
        description="Compute delay laws for a 2D array at a fluid/solid interface in 3D.",
        epilog=(
            "Example usage:\n"
            "  1) With plot:\n"
            "     python interface/delay_laws3Dint_interface.py --Mx=4 --My=4 --sx=0.5 --sy=0.5 --theta=20 --phi=0 --theta20=45 --DT0=10 --DF=10 --c1=1480 --c2=5900 --plt=y\n\n"
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
    
    args = parser.parse_args()
    
    # Compute delays
    td = run_delay_laws3Dint_service(args.Mx, args.My, args.sx, args.sy,
                                     args.theta, args.phi, args.theta20,
                                     args.DT0, args.DF, args.c1, args.c2,
                                     args.plt)
    
    print("Computed delays (in microseconds):")
    print(td)
    
if __name__ == "__main__":
    main()

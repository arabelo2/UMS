# interface/pts_2Dintf_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from application.pts_2Dintf_service import Pts2DIntfService

def parse_list(s: str):
    """Parse a comma‐separated string of numbers into a list of floats."""
    return [float(item) for item in s.split(',')]

def main():
    parser = argparse.ArgumentParser(
        description="Compute the intersection points xi on a plane interface for 2D array elements."
    )
    parser.add_argument("e", type=float, nargs="?", default=0.0,
                        help="Element offset from array center (mm) (default: 0.0)")
    parser.add_argument("xn", type=float, nargs="?", default=0.0,
                        help="Segment offset from element center (mm) (default: 0.0)")
    parser.add_argument("angt", type=float, nargs="?", default=0.0,
                        help="Array angle with respect to x-axis (degrees) (default: 0.0)")
    parser.add_argument("Dt0", type=float, nargs="?", default=10.0,
                        help="Distance from array center to interface (mm) (default: 10.0)")
    parser.add_argument("c1", type=float, nargs="?", default=1500.0,
                        help="Wave speed in medium 1 (m/s) (default: 1500.0)")
    parser.add_argument("c2", type=float, nargs="?", default=1500.0,
                        help="Wave speed in medium 2 (m/s) (default: 1500.0)")
    parser.add_argument("x", type=str, nargs="?", default="0",
                        help="x coordinate(s) (mm), comma separated if more than one (default: '0')")
    parser.add_argument("z", type=str, nargs="?", default="10",
                        help="z coordinate(s) (mm), comma separated if more than one (default: '10')")
    args = parser.parse_args()

    # Parse the x and z arguments.
    x_list = parse_list(args.x)
    z_list = parse_list(args.z)
    
    # If only one value is provided, treat it as scalar; otherwise, use list.
    x_val = x_list[0] if len(x_list) == 1 else x_list
    z_val = z_list[0] if len(z_list) == 1 else z_list

    service = Pts2DIntfService(args.e, args.xn, args.angt, args.Dt0,
                                args.c1, args.c2, x_val, z_val)
    xi = service.compute()
    print("Intersection points xi:")
    print(xi)

if __name__ == "__main__":
    main()

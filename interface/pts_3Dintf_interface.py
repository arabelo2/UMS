# interface/pts_3Dintf_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from application.pts_3Dintf_service import run_pts_3Dintf_service

def main():
    parser = argparse.ArgumentParser(
        description="Compute 3D intersection distances for a fluid/solid interface.",
        epilog="Example: python interface/pts_3Dintf_interface.py --ex 0 --ey 0 --xn 1 --yn 1 --angt 30 --Dt0 100 --c1 1480 --c2 5900 --x 50 --y 50 --z 80"
    )
    
    parser.add_argument("--ex", type=float, default=0, help="Element x-offset (mm).")
    parser.add_argument("--ey", type=float, default=0, help="Element y-offset (mm).")
    parser.add_argument("--xn", type=float, default=1, help="Segment x-offset (mm).")
    parser.add_argument("--yn", type=float, default=1, help="Segment y-offset (mm).")
    parser.add_argument("--angt", type=float, default=30, help="Array angle (degrees).")
    parser.add_argument("--Dt0", type=float, default=100, help="Array distance to interface (mm).")
    parser.add_argument("--c1", type=float, default=1480, help="Wave speed in first medium (m/s).")
    parser.add_argument("--c2", type=float, default=5900, help="Wave speed in second medium (m/s).")
    parser.add_argument("--x", type=float, default=50, help="X-coordinate (mm).")
    parser.add_argument("--y", type=float, default=50, help="Y-coordinate (mm).")
    parser.add_argument("--z", type=float, default=80, help="Z-coordinate (mm).")

    args = parser.parse_args()

    xi = run_pts_3Dintf_service(args.ex, args.ey, args.xn, args.yn, args.angt, args.Dt0, args.c1, args.c2, args.x, args.y, args.z)
    # If xi is a single scalar, format it, otherwise print the full array
    if xi.size == 1:
        print(f"Computed intersection point xi: {xi.item():.6f} mm")
    else:
        print(f"Computed intersection points xi:\n{xi}")

if __name__ == "__main__":
    main()

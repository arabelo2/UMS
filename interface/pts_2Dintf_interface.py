# interface/pts_2Dintf_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from application.pts_2Dintf_service import run_pts_2Dintf_service

def main():
    parser = argparse.ArgumentParser(
        description="pts_2Dintf Simulation Interface",
        epilog=(
            "Example usage:\n"
            "  python pts_2Dintf_interface.py --e 0 --xc 1.5 --angt 45 --Dt0 100 --c1 1480 --c2 5900 --x 50 --z 80\n"
            "This computes the intersection point (xi) in mm for the given parameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--e", type=float, default=0, help="Element offset from array center (mm).")
    parser.add_argument("--xc", type=float, default=1.0, help="Segment offset from element center (mm).")
    parser.add_argument("--angt", type=float, default=45, help="Angle of the array (degrees).")
    parser.add_argument("--Dt0", type=float, default=100, help="Distance from the array center to the interface (mm).")
    parser.add_argument("--c1", type=float, default=1480, help="Wave speed in medium one (m/s).")
    parser.add_argument("--c2", type=float, default=5900, help="Wave speed in medium two (m/s).")
    parser.add_argument("--x", type=float, default=50, help="x-coordinate of the observation point (mm).")
    parser.add_argument("--z", type=float, default=80, help="z-coordinate of the observation point (mm).")
    
    args = parser.parse_args()
    
    xi = run_pts_2Dintf_service(args.e, args.xc, args.angt, args.Dt0, args.c1, args.c2, args.x, args.z)
    print(f"Computed intersection point xi: {xi:.6f} mm")

if __name__ == '__main__':
    main()

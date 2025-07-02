# interface/pts_3Dintf_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from application.pts_3Dintf_service import run_pts_3Dintf_service

# Default parameters matching Ps3DInt
DEFAULTS = {
    'ex': 0,
    'ey': 0,
    'xn': 1,
    'yn': 1,
    'angt': 0,
    'Dt0': 10,
    'c1': 1480,
    'c2': 6320,  # Matches cp2 (longitudinal wave speed in solid)
    'x': 50,
    'y': 50,
    'z': 80
}

def main():
    parser = argparse.ArgumentParser(
        description="Compute intersection point xi for a fluid-solid interface using default Ps3DInt parameters.",
        epilog="Example: python interface/pts_3Dintf_interface.py --x 50 --y 50 --z 80"
    )

    parser.add_argument("--ex", type=float, default=DEFAULTS['ex'], help="Element x-offset (mm).")
    parser.add_argument("--ey", type=float, default=DEFAULTS['ey'], help="Element y-offset (mm).")
    parser.add_argument("--xn", type=float, default=DEFAULTS['xn'], help="Segment x-offset (mm).")
    parser.add_argument("--yn", type=float, default=DEFAULTS['yn'], help="Segment y-offset (mm).")
    parser.add_argument("--angt", type=float, default=DEFAULTS['angt'], help="Array angle (degrees).")
    parser.add_argument("--Dt0", type=float, default=DEFAULTS['Dt0'], help="Distance to interface (mm).")
    parser.add_argument("--c1", type=float, default=DEFAULTS['c1'], help="Wave speed in first medium (m/s).")
    parser.add_argument("--c2", type=float, default=DEFAULTS['c2'], help="Wave speed in second medium (m/s).")
    parser.add_argument("--x", type=float, default=DEFAULTS['x'], help="X-coordinate (mm).")
    parser.add_argument("--y", type=float, default=DEFAULTS['y'], help="Y-coordinate (mm).")
    parser.add_argument("--z", type=float, default=DEFAULTS['z'], help="Z-coordinate (mm).")

    args = parser.parse_args()

    # Call the application layer explicitly
    xi = run_pts_3Dintf_service(
        ex=args.ex, 
        ey=args.ey, 
        xn=args.xn, 
        yn=args.yn, 
        angt=args.angt, 
        Dt0=args.Dt0, 
        c1=args.c1, 
        c2=args.c2, 
        x=args.x, 
        y=args.y, 
        z=args.z
    )

    # Corrected handling to avoid AttributeError
    if isinstance(xi, (float, int, np.floating)):
        print(f"Computed intersection point xi: {xi:.6f} mm")
    else:
        print(f"Computed intersection points xi:\n{xi}")

if __name__ == "__main__":
    main()


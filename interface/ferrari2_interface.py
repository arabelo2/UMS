# interface/ferrari2_interface.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from application.ferrari2_service import Ferrari2Service

def main():
    """
    Parses command-line arguments, creates the Ferrari2 service,
    computes the intersection point, and prints the result.
    """
    parser = argparse.ArgumentParser(
        description="Solve for the interface intersection point xi using Ferrari's method."
    )
    parser.add_argument(
        "cr", type=float, nargs="?", default=1.5,
        help="Wave speed ratio c1/c2 (default: 1.5)"
    )
    parser.add_argument(
        "DF", type=float, nargs="?", default=10.0,
        help="Depth of the point in medium two (DF) (default: 10.0)"
    )
    parser.add_argument(
        "DT", type=float, nargs="?", default=5.0,
        help="Height of the point in medium one (DT) (default: 5.0)"
    )
    parser.add_argument(
        "DX", type=float, nargs="?", default=20.0,
        help="Separation distance between points (DX) (default: 20.0)"
    )
    args = parser.parse_args()

    service = Ferrari2Service(args.cr, args.DF, args.DT, args.DX)
    xi = service.solve()
    print(f"Intersection point xi: {xi}")

if __name__ == "__main__":
    main()

# interface/interface2_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from application.interface2_service import Interface2Service
from domain.interface2 import Interface2Parameters

def main():
    """
    Parses command-line arguments, creates the service, computes the function value,
    and prints the result.
    """
    parser = argparse.ArgumentParser(description="Compute interface2 function value")
    parser.add_argument("x", type=float, nargs="?", default=5.0, help="Location along the interface (default: 5.0)")
    parser.add_argument("cr", type=float, nargs="?", default=1.5, help="Wave speed ratio c1/c2 (default: 1.5)")
    parser.add_argument("df", type=float, nargs="?", default=10.0, help="Depth of the point in medium two (DF) (default: 10.0)")
    parser.add_argument("dp", type=float, nargs="?", default=5.0, help="Height of the point in medium one (DT) (default: 5.0)")
    parser.add_argument("dpf", type=float, nargs="?", default=20.0, help="Separation distance between points (DX) (default: 20.0)")
    args = parser.parse_args()

    # Build the domain parameters from the input arguments.
    parameters = Interface2Parameters(cr=args.cr, df=args.df, dp=args.dp, dpf=args.dpf)
    
    # Create the service with the provided parameters.
    service = Interface2Service(parameters)
    
    # Compute and output the result.
    result = service.compute(args.x)
    print(f"Computed value: {result}")

if __name__ == "__main__":
    main()

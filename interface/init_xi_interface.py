"""
Module: init_xi_interface.py
Layer: Interface

Provides a command-line interface (CLI) to test the init_xi functionality.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from application.init_xi_service import InitXiService
import numpy as np

def parse_vector(s: str):
    """
    Convert a comma-separated string of numbers into a list of floats.
    """
    try:
        return [float(item) for item in s.split(',')]
    except Exception:
        raise argparse.ArgumentTypeError("Must be a comma-separated list of numbers.")

def main():
    parser = argparse.ArgumentParser(
        description="Initialize the xi array based on x and z inputs."
    )

    parser.add_argument("--x", type=str, nargs="?", default="1,2,3",
                        help="x values as comma separated numbers (or a single number). Default: '1,2,3'")
    parser.add_argument("--z", type=str, nargs="?", default="10",
                        help="z values as comma separated numbers (or a single number). Default: '10'")
    args = parser.parse_args()

    # Parse input strings.
    x_list = parse_vector(args.x)
    z_list = parse_vector(args.z)

    # If a single value is provided, treat it as scalar.
    x_val = x_list[0] if len(x_list) == 1 else x_list
    z_val = z_list[0] if len(z_list) == 1 else z_list

    service = InitXiService(x_val, z_val)
    xi, P, Q = service.compute()

    print("Initialized xi array shape:", xi.shape)
    print("P =", P, "Q =", Q)
    print("xi:")
    print(xi)

if __name__ == "__main__":
    main()

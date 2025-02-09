# interface/rs_2Dv_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import ast
import operator as op
from application.rs_2Dv_service import run_rs_2Dv_service

# Supported operators for safe arithmetic evaluation.
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg
}

def _eval(node):
    """
    Recursively evaluate an AST node.
    Uses ast.Constant for numbers (Python 3.8+) and falls back to raising an error for any unsupported expression.
    """
    if isinstance(node, ast.Constant):  # For Python 3.8+ (replaces ast.Num)
        return node.value
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval(node.operand))
    else:
        raise TypeError(f"Unsupported expression: {node}")

def safe_eval(expr):
    """
    Safely evaluate a simple arithmetic expression using ast.
    Only numbers and basic arithmetic operators are allowed.
    """
    try:
        return _eval(ast.parse(expr, mode='eval').body)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid arithmetic expression: {expr}") from e

def safe_float(s: str) -> float:
    """
    Convert a string to float.
    If direct conversion fails, attempt to safely evaluate it as an arithmetic expression.
    """
    try:
        return float(s)
    except ValueError:
        return safe_eval(s)

def parse_z(z_str: str):
    """
    Parse a comma-separated string into a NumPy array.
    
    - If exactly three numbers are provided, they are interpreted as
      [start, stop, num_points] and used to generate z via np.linspace.
    - Otherwise, the numbers are interpreted as explicit z-values.
    
    Extra surrounding quotes (both single and double) are stripped.
    
    :param z_str: String containing comma-separated numbers.
    :return: NumPy array of floats.
    """
    # Remove any extra surrounding quotes.
    z_str = z_str.strip("'\"")
    try:
        values = [safe_float(val.strip()) for val in z_str.split(',')]
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid format for --z. It should be a comma-separated list of numbers, e.g. '5,200,500' for linspace or '5,10,15,20' for explicit values."
        )
    
    if len(values) == 3:
        # Interpret as linspace parameters: start, stop, and number of points.
        start, stop, num = values
        return np.linspace(start, stop, int(num))
    else:
        # Treat as explicit z-values.
        return np.array(values)

def main():
    parser = argparse.ArgumentParser(
        description="Rayleigh–Sommerfeld 2-D Simulation Interface",
        epilog=(
            "Example usage with arithmetic expressions:\n"
            "  python rs_2Dv_interface.py --b 6.35/2 --f 5 --c 1500 --e 0 --x 0 --z '1,50,500' --N 50\n\n"
            "If --z is not provided, the default is 500 points linearly spaced between 5 and 200 mm.\n"
            "If exactly three numbers are provided for --z (e.g., '5,200,500'), they are interpreted as start, stop, and number of points.\n"
            "Otherwise, the provided numbers are taken as explicit z coordinates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--b", type=safe_float, default=6.35/2,
                        help="Half-length of the element (mm). Default: 6.35/2 (≈3.175 mm)")
    parser.add_argument("--f", type=safe_float, default=5,
                        help="Frequency (MHz). Default: 5")
    parser.add_argument("--c", type=safe_float, default=1500,
                        help="Wave speed (m/s). Default: 1500")
    parser.add_argument("--e", type=safe_float, default=0,
                        help="Lateral offset of the element's center (mm). Default: 0")
    parser.add_argument("--x", type=safe_float, default=0,
                        help="x-coordinate (mm) where pressure is computed. Default: 0")
    parser.add_argument("--z", type=str, default=None,
                        help=(
                              "z-coordinate(s) (mm) where pressure is computed. "
                              "Provide a comma-separated list of numbers. If exactly three numbers are provided, they are interpreted as start, stop, and number of points for np.linspace (e.g., '5,200,500'). "
                              "Otherwise, the numbers are taken as explicit z-values. "
                              "Default: 500 points linearly spaced between 5 and 200 mm."
                              ))
    parser.add_argument("--N", type=int, default=None,
                        help="Optional number of segments for numerical integration. Default: computed automatically.")
    
    args = parser.parse_args()

    # Process the z-coordinate: if not provided, use the default linspace.
    try:
        if args.z is None:
            z = np.linspace(5, 200, 500)
        else:
            z = parse_z(args.z)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    
    # Call the service layer (which calls the domain code).
    p = run_rs_2Dv_service(args.b, args.f, args.c, args.e, args.x, z, args.N)

    # Display the computed pressure.
    if isinstance(p, np.ndarray):
        print("Computed normalized pressure (magnitude) for each z value:")
        print(np.abs(p))
    else:
        print("Computed normalized pressure (magnitude):")
        print(abs(p))
    
    # Generate the plot: plot(z, abs(p))
    plt.figure(figsize=(8, 5))
    plt.plot(z, np.abs(p), 'b-', lw=2)
    plt.xlabel("z (mm)")
    plt.ylabel("Normalized Pressure Magnitude")
    plt.title("Rayleigh–Sommerfeld 2-D Simulation")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

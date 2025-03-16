# interface/cli_utils.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import operator as op
import argparse
import numpy as np

# Supported operators for safe arithmetic evaluation.
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.pow,  # Treat '^' as exponentiation (MATLAB style)
    ast.USub: op.neg
}

def _eval(node):
    """
    Recursively evaluate an AST node.
    Uses ast.Constant for Python 3.8+.
    """
    if isinstance(node, ast.Constant):
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
    
    Example usage:
      safe_eval("6.35/2") -> 3.175
      safe_eval("10+5")   -> 15
    """
    try:
        return _eval(ast.parse(expr, mode='eval').body)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid arithmetic expression: {expr}") from e

def safe_float(s: str) -> float:
    """
    Convert a string to float.
    If direct conversion fails, attempt to safely evaluate it as an arithmetic expression.
    Special handling: 'inf' or 'infinite' (case-insensitive) are converted to float('inf').
    
    Examples:
      safe_float("3.14")      -> 3.14
      safe_float("6.35/2")    -> 3.175
      safe_float("inf")       -> float('inf')
      safe_float("infinite")  -> float('inf')
      safe_float("invalid")   -> raises argparse.ArgumentTypeError
    """
    s_lower = s.lower().strip()
    if s_lower in ['inf', 'infinite']:
        return float('inf')
    try:
        return float(s)
    except ValueError:
        return safe_eval(s)

def parse_array(arr_str: str):
    """
    Parse a comma-separated string into a NumPy array.
    
    - If exactly three numbers are provided, they are interpreted as
      [start, stop, num_points] and used to generate an array via np.linspace.
      For example, "5,80,200" -> np.linspace(5,80,200).
    - Otherwise, the numbers are treated as explicit values,
      e.g. "5,10,15,20" -> np.array([5,10,15,20]).
    - Negative values for start or stop are supported by enclosing the argument in quotes,
      e.g., --x="-10,10,200".
    
    Extra surrounding quotes are stripped if present.
    
    Raises:
      argparse.ArgumentTypeError if the string cannot be parsed into valid numbers.
    
    Returns:
      A NumPy array of floats.
    """
    arr_str = arr_str.strip("'\"")
    try:
        values = [safe_float(val.strip()) for val in arr_str.split(',')]
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid format. Should be a comma-separated list of numbers, e.g. '5,200,500' for linspace "
            "or '5,10,15,20' for explicit values. For negative values, use quotes, e.g. --x=\"-10,10,200\"."
        )
    if len(values) == 3:
        start, stop, num = values
        return np.linspace(start, stop, int(num))
    else:
        return np.array(values)

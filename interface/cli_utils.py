# interface/cli_utils.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import operator as op
import argparse
import numpy as np  # Make sure NumPy is imported globally

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

def parse_array(arr_str: str):
    """
    Parse a comma-separated string into a NumPy array.
    
    - If exactly three numbers are provided, they are interpreted as
      [start, stop, num_points] and used to generate an array via np.linspace.
    - Otherwise, the numbers are treated as explicit values.
    
    Extra surrounding quotes are stripped.
    """
    arr_str = arr_str.strip("'\"")
    try:
        values = [safe_float(val.strip()) for val in arr_str.split(',')]
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid format. Should be a comma-separated list of numbers, e.g. '5,200,500' for linspace or '5,10,15,20' for explicit values."
        )
    if len(values) == 3:
        start, stop, num = values
        return np.linspace(start, stop, int(num))
    else:
        return np.array(values)

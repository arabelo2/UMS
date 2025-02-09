import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from domain.ferrari2 import FerrariSolver

coefficients = [1, -4, 6, -4, 1]  # Corresponds to (x - 1)^4 = 0
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 1 - Expected: [1, 1, 1, 1], Computed:", roots)


coefficients = [1, 0, 2, 0, 1]
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 2 - Expected: [±i, ±i], Computed:", roots)

coefficients = [1, -2, 5, -2, 2]
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 3 - Computed Roots:", roots)


coefficients = [0, 1, -3, 3, -1]  # Becomes x^3 - 3x^2 + 3x - 1 = 0
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 4 - Expected: [1, 1, 1], Computed:", roots)

coefficients = [0, 0, 1, -4, 3]  # Becomes x^2 - 4x + 3 = 0
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 5 - Expected: [3, 1], Computed:", roots)


coefficients = [1, -1000, 100000, -1000000, 10000000]
solver = FerrariSolver(coefficients)
roots = solver.solve()
print("Test 6 - Large Coefficients, Computed Roots:", roots)


import numpy as np

for _ in range(5):  # Run 5 random tests
    coefficients = np.random.randint(-10, 10, size=5)
    solver = FerrariSolver(coefficients)
    my_roots = solver.solve()
    numpy_roots = np.roots(coefficients)
    
    print(f"Test 7 - Coefficients: {coefficients}")
    print("  FerrariSolver Roots:", my_roots)
    print("  NumPy Roots:", numpy_roots)


# application/init_xi_service.py

"""
Module: init_xi_service.py
Layer: Application

Provides a service that wraps the domain solver (InitXiSolver) for initializing xi.
"""

from domain.init_xi import InitXiSolver
from typing import Tuple, Any
import numpy as np


class InitXiService:
    """
    Service class that uses InitXiSolver to compute the initialized array xi.
    """

    def __init__(self, x: Any, z: Any):
        self.solver = InitXiSolver(x, z)

    def compute(self) -> Tuple[np.ndarray, int, int]:
        """
        Compute xi and its dimensions.

        Returns:
            xi (np.ndarray): The initialized array.
            P (int): Number of rows.
            Q (int): Number of columns.
        """
        return self.solver.compute()

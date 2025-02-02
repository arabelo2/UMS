from domain.discrete_windows import DiscreteWindows
import numpy as np

class DiscreteWindowsService:
    """Service layer for computing discrete windowing functions."""

    def __init__(self, M: int, window_type: str):
        self.M = M
        self.window_type = window_type
        self.weights = None

    def calculate_weights(self) -> np.ndarray:
        """Compute and return the window weights."""
        self.weights = DiscreteWindows.generate_weights(self.M, self.window_type)
        return self.weights

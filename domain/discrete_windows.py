import numpy as np

class DiscreteWindows:
    """Computes discrete apodization amplitudes for an array of M elements."""

    @staticmethod
    def generate_weights(M: int, window_type: str) -> np.ndarray:
        """
        Generate apodization weights for M elements based on the selected window type.

        Parameters:
            M (int): Number of elements in the array.
            window_type (str): Type of windowing function ('cos', 'Han', 'Ham', 'Blk', 'tri', 'rect').

        Returns:
            np.ndarray: Apodization weights for M elements.
        """
        if M <= 1:
            raise ValueError("M must be greater than 1 for windowing to be meaningful.")

        m = np.arange(M)  # Element indices from 0 to M-1
        factor = np.pi * m / (M - 1)  # Common factor used in multiple window types

        window_types = {
            "cos": np.sin(factor),
            "Han": np.sin(factor) ** 2,
            "Ham": 0.54 - 0.46 * np.cos(2 * factor),
            "Blk": 0.42 - 0.5 * np.cos(2 * factor) + 0.08 * np.cos(4 * factor),
            "tri": 1 - np.abs(2 * m / (M - 1) - 1),
            "rect": np.ones(M),
        }

        if window_type not in window_types:
            raise ValueError("Invalid window type. Choose from 'cos', 'Han', 'Ham', 'Blk', 'tri', 'rect'.")

        return window_types[window_type]

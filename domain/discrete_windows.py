import numpy as np


class DiscreteWindows:
    """
    Computes discrete apodization amplitudes for various window types.
    """

    @staticmethod
    def compute(M, window_type):
        """
        Compute the discrete apodization amplitudes.

        Parameters:
            M (int): Number of elements.
            window_type (str): Type of window ('cos', 'Han', 'Ham', 'Blk', 'tri', 'rect').

        Returns:
            numpy.ndarray: Array of apodization amplitudes.
        """
        m = np.arange(1, M + 1)  # Element indices

        if window_type == 'cos':
            amp = np.sin(np.pi * (m - 1) / (M - 1))
        elif window_type == 'Han':
            amp = np.sin(np.pi * (m - 1) / (M - 1))**2
        elif window_type == 'Ham':
            amp = 0.54 - 0.46 * np.cos(2 * np.pi * (m - 1) / (M - 1))
        elif window_type == 'Blk':
            amp = (0.42 - 0.5 * np.cos(2 * np.pi * (m - 1) / (M - 1)) +
                   0.08 * np.cos(4 * np.pi * (m - 1) / (M - 1)))
        elif window_type == 'tri':
            amp = 1 - np.abs(2 * (m - 1) / (M - 1) - 1)
        elif window_type == 'rect':
            amp = np.ones(M)
        else:
            raise ValueError(
                "Invalid window type. Choices are 'cos', 'Han', 'Ham', 'Blk', 'tri', 'rect'."
            )

        return amp

    @staticmethod
    def compute_weights(M, type_):
        """
        Compute amplitude weights for an array using a given window function.

        Parameters:
            M (int): Number of elements in the array.
            type_ (str): Type of window function ('rect', 'hamming', etc.).

        Returns:
            np.ndarray: Array of amplitude weights.
        """
        if type_.lower() == "rect":
            return np.ones(M)  # Rectangular window (uniform weighting)
        elif type_.lower() == "hamming":
            return np.hamming(M)  # Hamming window
        elif type_.lower() == "hanning":
            return np.hanning(M)  # Hanning window
        elif type_.lower() == "blackman":
            return np.blackman(M)  # Blackman window
        else:
            raise ValueError(f"Unsupported window type: {type_}")

# domain/discrete_windows.py

import numpy as np

def discrete_windows(M: int, wtype: str) -> np.ndarray:
    """
    Return the discrete apodization amplitudes for M elements of the given window type.
    
    The available types are:
      'cos' -> Cosine window
      'Han' -> Hanning window
      'Ham' -> Hamming window
      'Blk' -> Blackman window
      'tri' -> Triangular window
      'rect'-> Rectangular window (no apodization)
    
    Parameters:
      M (int)     : Number of discrete elements (must be >= 1).
      wtype (str) : The window type string.
    
    Returns:
      np.ndarray of shape (M,) with amplitude values.
    
    Raises:
      ValueError: if M < 1
      ValueError: if wtype is not one of the recognized types.
    
    Notes:
    - This function replicates the behavior of the MATLAB function discrete_windows.m.
    - We can reference the approach from Cormen for clarity: each array is generated
      in O(M) time, which is already optimal for constructing an output of size M.
    """
    if M < 1:
        raise ValueError("Number of elements M must be >= 1.")
    
    # Handle single element: simply return 1 for the only element.
    if M == 1:
        return np.ones(1)
    
    # Element indices (MATLAB code uses m=1:M)
    m = np.arange(1, M+1)
    
    # Generate the amplitude array using the specified window type.
    if wtype == 'cos':
        # amp = sin(pi*(m-1)/(M-1))
        amp = np.sin(np.pi * (m - 1) / (M - 1))
    elif wtype == 'Han':
        # amp = (sin(pi*(m-1)/(M-1)))^2
        amp = np.sin(np.pi * (m - 1) / (M - 1))**2
    elif wtype == 'Ham':
        # amp = 0.54 - 0.46*cos(2*pi*(m-1)/(M-1))
        amp = 0.54 - 0.46 * np.cos(2 * np.pi * (m - 1) / (M - 1))
    elif wtype == 'Blk':
        # amp = 0.42 - 0.5*cos(2*pi*(m-1)/(M-1)) + 0.08*cos(4*pi*(m-1)/(M-1))
        amp = (0.42
               - 0.5  * np.cos(2 * np.pi * (m - 1) / (M - 1))
               + 0.08 * np.cos(4 * np.pi * (m - 1) / (M - 1)))
    elif wtype == 'tri':
        # amp = 1 - |2*(m-1)/(M-1) - 1|
        amp = 1.0 - np.abs(2 * (m - 1) / (M - 1) - 1)
    elif wtype == 'rect':
        # amp = ones(1, M)
        amp = np.ones(M)
    else:
        # If type is invalid
        raise ValueError("Invalid window type. Choices are 'cos', 'Han', 'Ham', 'Blk', 'tri', 'rect'")

    return amp

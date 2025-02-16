# application/discrete_windows_service.py

from domain.discrete_windows import discrete_windows

def run_discrete_windows_service(M, wtype):
    """
    Service function to generate discrete window amplitudes for M elements.
    
    Parameters:
      M (int)       : number of elements (>=1)
      wtype (str)   : one of 'cos', 'Han', 'Ham', 'Blk', 'tri', or 'rect'
    
    Returns:
      amp (np.ndarray): The discrete amplitudes for the chosen window type.
    """
    return discrete_windows(M, wtype)

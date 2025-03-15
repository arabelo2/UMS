# application/delay_laws3Dint_service.py

from domain.delay_laws3Dint import delay_laws3Dint

def run_delay_laws3Dint_service(Mx: int, My: int, sx: float, sy: float,
                                thetat: float, phi: float, theta2: float,
                                DT0: float, DF: float, c1: float, c2: float,
                                plt_option: str = 'n'):
    """
    Service function to compute delay laws for a 2D array at a fluid/solid interface.
    
    Returns:
        td (np.ndarray): 2D array of time delays (in microseconds) with shape (Mx, My).
    """
    return delay_laws3Dint(Mx, My, sx, sy, thetat, phi, theta2, DT0, DF, c1, c2, plt_option)

# application/delay_laws3Dint_service.py

from domain.delay_laws3Dint import delay_laws3Dint

def run_delay_laws3Dint_service(Mx: int, My: int, sx: float, sy: float,
                                theta: float, phi: float, theta20: float,
                                DT0: float, DF: float, c1: float, c2: float,
                                plt_option: str = 'n'):
    """
    Service function to compute delay laws for a 2D array at a fluid/solid interface.
    
    Parameters:
        Mx, My   : int     - Number of elements in the (x', y') directions.
        sx, sy   : float   - Pitches (in mm) along the x' and y' directions.
        theta    : float   - Array angle with the interface (in degrees).
        phi      : float   - Steering parameter for the second medium (in degrees).
        theta20  : float   - Refracted steering angle in the second medium (in degrees).
        DT0      : float   - Height of the array center above the interface (mm).
        DF       : float   - Focal distance in the second medium (mm). Use np.inf for steering-only.
        c1, c2   : float   - Wave speeds (in m/s) for the first and second media, respectively.
        plt_option: str    - 'y' to plot the ray geometry, 'n' otherwise.
    
    Returns:
        td (np.ndarray): 2D array of time delays (in microseconds) with shape (Mx, My).
    """
    return delay_laws3Dint(Mx, My, sx, sy, theta, phi, theta20, DT0, DF, c1, c2, plt_option)

# application/delay_laws2D_service.py

from domain.delay_laws2D import delay_laws2D

def run_delay_laws2D_service(M, s, Phi, F, c):
    """
    Service function to compute delay laws for a 1-D array.

    Parameters:
        M (int)    : Number of elements in the array (should be odd).
        s (float)  : Pitch in mm.
        Phi (float): Steering angle in degrees.
        F (float)  : Focal distance in mm. np.inf => steering only.
        c (float)  : Wave speed (m/s).

    Returns:
        td (np.ndarray): Time delays (microseconds) for each element.
    """
    return delay_laws2D(M, s, Phi, F, c)

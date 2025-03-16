# application/delay_laws3D_service.py

from domain.delay_laws3D import DelayLaws3D

def run_delay_laws3D_service(M: int, N: int, sx: float, sy: float,
                             theta: float, phi: float, F: float, c: float):
    """
    Service function to compute delay laws for a 2D array.

    Parameters:
        M (int): Number of elements in x-direction.
        N (int): Number of elements in y-direction.
        sx (float): Pitch in x-direction (mm).
        sy (float): Pitch in y-direction (mm).
        theta (float): Steering angle theta (degrees).
        phi (float): Steering angle phi (degrees).
        F (float): Focal distance (mm). Use np.inf for steering only.
        c (float): Wave speed (m/s).

    Returns:
        np.ndarray: 2D array of time delays (microseconds).
    """
    delay_instance = DelayLaws3D(M, N, sx, sy, theta, phi, F, c)
    return delay_instance.calculate()

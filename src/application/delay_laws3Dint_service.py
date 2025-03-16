# application/delay_laws3Dint_service.py

from domain.delay_laws3Dint import delay_laws3Dint

def run_delay_laws3Dint_service(Mx: int, My: int, sx: float, sy: float,
                                theta: float, phi: float, theta20: float,
                                DT0: float, DF: float, c1: float, c2: float,
                                plt_option: str = 'n',
                                view_elev: float = 25.0, view_azim: float = 20.0,
                                z_scale: float = 1.0):
    """
    Service function to compute delay laws for a 2D array at a fluid/solid interface in 3D.
    
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
        view_elev: float   - Camera elevation for the 3D plot. Default: 25.
        view_azim: float   - Camera azimuth for the 3D plot. Default: 20.
        z_scale  : float    - Scale factor for the z-axis (delay values) for visualization. Default: 1.0.
        
    Returns:
        td       : np.ndarray
                   The physical 2D array of time delays (in microseconds) with shape (Mx, My).
        td_scaled: np.ndarray
                   The delay array scaled by z_scale (for plotting).
    """
    td = delay_laws3Dint(Mx, My, sx, sy, theta, phi, theta20, DT0, DF, c1, c2, plt_option, view_elev, view_azim)
    td_scaled = td * z_scale
    return td, td_scaled

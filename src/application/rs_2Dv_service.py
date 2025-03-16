# application/rs_2Dv_service.py

from domain.rs_2Dv import RayleighSommerfeld2Dv

def run_rs_2Dv_service(b: float, f: float, c: float, e: float, x, z, N: int = None):
    """
    Service function to run the Rayleigh–Sommerfeld 2-D simulation.
    
    :param b: Half-length of the element (mm).
    :param f: Frequency (MHz).
    :param c: Wave speed (m/s).
    :param e: Lateral offset (mm).
    :param x: x-coordinate (mm) where pressure is evaluated.
    :param z: z-coordinate (mm) or array of z-coordinates.
    :param N: (Optional) Number of segments for numerical integration.
    :return: Computed normalized pressure (complex or NumPy array).
    """
    simulator = RayleighSommerfeld2Dv(b, f, c, e)
    return simulator.compute_pressure(x, z, N)

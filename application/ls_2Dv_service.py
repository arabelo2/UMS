# application/ls_2Dv_service.py

from domain.ls_2Dv import LS2Dv

def run_ls_2Dv_service(b: float, f: float, c: float, e: float, x, z, N: int = None):
    """
    Service function to run the LS2Dv simulation.
    
    :param b: Half-length of the element (mm).
    :param f: Frequency (MHz).
    :param c: Wave speed (m/s).
    :param e: Lateral offset (mm).
    :param x: x-coordinate(s) (mm) for pressure evaluation.
    :param z: z-coordinate(s) (mm) for pressure evaluation.
    :param N: Optional number of segments; if not provided, computed automatically.
    :return: Computed normalized pressure (complex or NumPy array).
    """
    simulator = LS2Dv(b, f, c, e)
    return simulator.compute_pressure(x, z, N)

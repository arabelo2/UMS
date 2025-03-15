from domain.mps_array_model_int import MPSArrayModelInt

def run_mps_array_model_int_service(lx: float, ly: float, gx: float, gy: float,
                                    f: float, d1: float, c1: float, d2: float, c2: float, cs2: float, wave_type: str,
                                    L1: int, L2: int, angt: float, Dt0: float,
                                    theta20: float, phi: float, DF: float,
                                    ampx_type: str, ampy_type: str,
                                    xs, zs, y: float):
    """
    Service function to compute the normalized velocity field using MPSArrayModelInt.
    
    Returns:
        dict: Contains the velocity field ('p') and grid coordinates ('x', 'z').
    """
    model = MPSArrayModelInt(lx, ly, gx, gy,
                             f, d1, c1, d2, c2, cs2, wave_type,
                             L1, L2, angt, Dt0,
                             theta20, phi, DF,
                             ampx_type, ampy_type,
                             xs, zs, y)
    result = model.compute_field()
    if not isinstance(result, dict) or 'p' not in result or 'x' not in result or 'z' not in result:
        raise ValueError("Service did not return expected result dictionary with keys 'p', 'x', 'z'.")
    return result

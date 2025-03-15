# application/mps_array_modeling_service.py

from domain.mps_array_modeling import MPSArrayModeling

def run_mps_array_modeling_service(lx, ly, gx, gy, f, c, L1, L2,
                                   theta, phi, F, ampx_type, ampy_type,
                                   xs=None, zs=None, y=0.0):
    """
    Service function to compute the normalized pressure field.

    Parameters:
      (See the domain class for full parameter descriptions.)

    Returns:
       p : np.ndarray (complex)
          Normalized pressure field.
       xs, zs : np.ndarray
          Evaluation grid coordinates.
    """
    model = MPSArrayModeling(lx, ly, gx, gy, f, c, L1, L2,
                             theta, phi, F, ampx_type, ampy_type,
                             xs, zs, y)
    return model.compute_pressure_field()

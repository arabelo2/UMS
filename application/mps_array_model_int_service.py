# application/mps_array_model_int_service.py

from domain.mps_array_model_int import MPSArrayModelInt

class MLSArrayModelIntService:
    """
    Service layer for MLS Array Modeling at a fluid/solid interface.
    """
    def __init__(self, f, d1, c1, d2, c2, cs2, wave_type, M, d, g, angt, ang20, DF, DT0, wtype, x=None, z=None):
        self.model = MPSArrayModelInt(f, d1, c1, d2, c2, cs2, wave_type, M, d, g, angt, DT0, ang20, x, z)

    def run(self):
        result = self.model.compute_field()
        if not isinstance(result, dict):
            raise ValueError("compute_field did not return a dictionary.")
        required_keys = ['p', 'x', 'z']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key '{key}' in result from compute_field.")
        return result

def run_mps_array_model_int_service(f, d1, c1, d2, c2, cs2, wave_type, M, d, g, angt, DT0, ang20, DF, wtype, x=None, z=None):
    service = MLSArrayModelIntService(f, d1, c1, d2, c2, cs2, wave_type, M, d, g, angt, DT0, ang20, DF, wtype, x, z)
    return service.run()

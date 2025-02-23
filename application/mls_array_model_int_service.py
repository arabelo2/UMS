# application/mls_array_model_int_service.py

from domain.mls_array_model_int import MLSArrayModelInt

class MLSArrayModelIntService:
    """
    Service layer for MLS Array Modeling at a fluid/fluid interface.
    """
    def __init__(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, x=None, z=None):
        self.model = MLSArrayModelInt(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, x, z)
    
    def run(self):
        return self.model.compute_field()

def run_mls_array_model_int_service(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, x=None, z=None):
    service = MLSArrayModelIntService(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, x, z)
    return service.run()

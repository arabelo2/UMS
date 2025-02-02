from domain.mls_array_model_int import MLSArrayModelInt

class MLSArrayModelIntService:
    """Service layer for computing 1D phased array interactions."""

    def __init__(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, window_type):
        self.solver = MLSArrayModelInt(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, window_type)

    def compute_pressure(self, x, z):
        """Compute pressure field."""
        return self.solver.compute_pressure_field(x, z)

from domain.on_axis_foc2D import OnAxisFocusedPiston

class OnAxisFocusedPistonService:
    """Service layer for on-axis focused piston modeling."""

    def __init__(self, b, R, f, c):
        self.solver = OnAxisFocusedPiston(b, R, f, c)

    def compute_pressure(self, z):
        """Compute pressure at depth `z`."""
        return self.solver.compute_pressure(z)

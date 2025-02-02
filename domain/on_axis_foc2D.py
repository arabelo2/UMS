from application.fresnel_int_service import FresnelIntegralService

class OnAxisFocusedPiston:
    def __init__(self, b, R, f, c):
        self.b = b
        self.R = R
        self.f = f
        self.c = c
        self.kb = (2000 * np.pi * f * b) / c  # Compute wave number

        # Initialize the service
        self.fresnel_service = FresnelIntegralService()

    def compute_pressure(self, z):
        """Compute the on-axis normalized pressure at a given depth `z`."""

        z = np.atleast_1d(z)  # Ensure z is an array
        z = z + np.finfo(float).eps * (z == 0)  # Avoid division by zero
        
        # Compute `u` parameter for near/far field
        u = 1 - z / self.R
        u = u + np.finfo(float).eps * (u == 0)  # Prevent singularities

        # Compute argument `x` of the Fresnel integral
        x_near = np.sqrt((u * self.kb * self.b) / (np.pi * z)) * (z <= self.R)
        x_far = np.sqrt((-u * self.kb * self.b) / (np.pi * z)) * (z > self.R)
        x = x_near + x_far

        # Compute denominator
        denom = np.sqrt(u) * (z <= self.R) + np.sqrt(-u) * (z > self.R)

        # Compute Fresnel integrals using the service
        fresnel_near = self.fresnel_service.compute_integrals(x) * (z <= self.R)
        fresnel_far = np.conj(self.fresnel_service.compute_integrals(x)) * (z > self.R)
        Fr = fresnel_near + fresnel_far

        # Compute final normalized pressure
        p_near = np.sqrt(2 / 1j) * np.sqrt((self.b / self.R) * self.kb / np.pi) * (np.abs(u) <= 0.005)
        p_far = np.sqrt(2 / 1j) * Fr / denom * (np.abs(u) > 0.005)
        p = p_near + p_far

        return p

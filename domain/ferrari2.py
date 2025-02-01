import numpy as np
from scipy.optimize import root_scalar
from domain.interface_function import InterfaceFunction


class FerrariSolver:
    """
    Solves for the intersection point, xi, on a plane interface along a Snell's law ray path.
    """

    def __init__(self, cr, DF, DT, DX):
        """
        Initialize the solver with given parameters.

        Parameters:
            cr (float): Wave speed ratio (c1/c2).
            DF (float): Depth of the point in medium two.
            DT (float): Height of the point in medium one.
            DX (float): Separation distance between points.
        """
        self.cr = cr
        self.DF = DF
        self.DT = DT
        self.DX = DX

    def solve(self):
        """
        Solve for the intersection point xi.

        Returns:
            float: Intersection point xi.
        """
        # Special case for identical media
        if abs(self.cr - 1) < 1e-6:
            return self.DX * self.DT / (self.DF + self.DT)

        # Calculate quartic coefficients
        cri = 1 / self.cr
        A = 1 - cri**2
        B = (2 * cri**2 * self.DX - 2 * self.DX) / self.DT
        C = (self.DX**2 + self.DT**2 - cri**2 * (self.DX**2 + self.DF**2)) / self.DT**2
        D = -2 * self.DX * self.DT**2 / self.DT**3
        E = self.DX**2 * self.DT**2 / self.DT**4

        # Solve using Ferrari's method
        alpha = -3 * B**2 / (8 * A**2) + C / A
        beta = B**3 / (8 * A**3) - B * C / (2 * A**2) + D / A
        gamma = -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A

        roots = self._solve_quartic(A, B, alpha, beta, gamma)
        xi = self._find_valid_root(roots)

        if xi is None:
            xi = self._fallback_to_numeric()

        return xi

    def _solve_quartic(self, A, B, alpha, beta, gamma):
        """
        Solve the quartic equation using Ferrari's method.

        Returns:
            list: Roots of the quartic equation.
        """
        # ✅ Use NumPy array-safe comparisons
        if np.all(beta == 0):  # Checks if all elements in beta are zero
            roots = [
                -B / (4 * A) + np.sqrt(max(0, (-alpha + np.sqrt(alpha**2 - 4 * gamma)) / 2)),
                -B / (4 * A) + np.sqrt(max(0, (-alpha - np.sqrt(alpha**2 - 4 * gamma)) / 2)),
                -B / (4 * A) - np.sqrt(max(0, (-alpha + np.sqrt(alpha**2 - 4 * gamma)) / 2)),
                -B / (4 * A) - np.sqrt(max(0, (-alpha - np.sqrt(alpha**2 - 4 * gamma)) / 2)),
            ]
        else:
            P = -alpha**2 / 12 - gamma
            Q = -alpha**3 / 108 + alpha * gamma / 3 - beta**2 / 8
            Rm = Q / 2 - np.sqrt(Q**2 / 4 + P**3 / 27)
            U = np.cbrt(Rm)
            
            # ✅ Use NumPy condition to avoid ambiguous truth value error
            U_is_zero = np.isclose(U, 0)

            y = np.where(
                U_is_zero, 
                -5 / 6 * alpha, 
                -5 / 6 * alpha - U + P / (3 * U)
            )
            
            W = np.sqrt(max(0, alpha + 2 * y))

            roots = [
                -B / (4 * A) + 0.5 * (W + np.sqrt(max(0, -(3 * alpha + 2 * y + 2 * beta / W)))) if np.all(beta != 0) else -B / (4 * A) + 0.5 * W,
                -B / (4 * A) + 0.5 * (-W + np.sqrt(max(0, -(3 * alpha + 2 * y - 2 * beta / W)))) if np.all(beta != 0) else -B / (4 * A) - 0.5 * W,
                -B / (4 * A) + 0.5 * (W - np.sqrt(max(0, -(3 * alpha + 2 * y + 2 * beta / W)))) if np.all(beta != 0) else -B / (4 * A) + 0.5 * W,
                -B / (4 * A) + 0.5 * (-W - np.sqrt(max(0, -(3 * alpha + 2 * y - 2 * beta / W)))) if np.all(beta != 0) else -B / (4 * A) - 0.5 * W,
            ]
        return roots

    def _find_valid_root(self, roots):
        """
        Find the valid root within the interval [0, DX].
        """
        tolerance = 1e-6
        for root in roots:
            real_part = np.real(root)
            imaginary_part = np.abs(np.imag(root))
            scaled_root = real_part * self.DT

            if self.DX >= 0 and 0 <= scaled_root <= self.DX and imaginary_part < tolerance:
                return scaled_root
            elif self.DX < 0 and self.DX <= scaled_root <= 0 and imaginary_part < tolerance:
                return scaled_root
        return None

    def _fallback_to_numeric(self):
        """
        Fallback to numerical solver if Ferrari's method fails.
        """
        def interface2_wrapper(x):
            interface = InterfaceFunction(self.cr, self.DF, self.DT, self.DX)
            return interface.calculate_y(x)

        result = root_scalar(interface2_wrapper, bracket=[0, self.DX])
        return result.root

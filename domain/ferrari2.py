# domain/ferrari2.py

import cmath
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(divide='ignore')  # This line tells NumPy to ignore divide-by-zero warnings.

import math
from math import sqrt

class Ferrari2:
    """
    Domain class to solve for the intersection point, xi, along an interface
    using Ferrari's method for the quartic obtained by writing Snell's law.
    
    Given:
        cr  : c1/c2, the ratio of wave speeds
        DF  : Depth of the point in medium two (must be positive)
        DT  : Height of the point in medium one (must be positive)
        DX  : Separation distance between the two points (can be positive or negative)
    
    The solution xi is defined as: xi = x * DT, where x is the root
    computed from the quartic. If Ferrari's method fails to yield an acceptable
    root, a numerical root finder is used as fallback.
    """
    TOL = 1e-8

    def __init__(self, cr: float, DF: float, DT: float, DX: float):
        self.cr = cr
        self.DF = DF
        self.DT = DT
        self.DX = DX

    def solve(self) -> float:
        """
        Solve for the intersection point xi.
        
        Returns:
            float: The intersection point xi.
        """
        # Special case: if media are identical, use explicit formula.
        if abs(self.cr - 1) < self.TOL:
            return self.DX * self.DT / (self.DF + self.DT)

        cri = 1 / self.cr  # c2/c1
        DT = self.DT
        DF = self.DF
        DX = self.DX

        # Define coefficients of quartic: A*x^4 + B*x^3 + C*x^2 + D*x + E = 0
        A = 1 - cri**2
        B = (2 * cri**2 * DX - 2 * DX) / DT
        C = (DX**2 + DT**2 - cri**2 * (DX**2 + DF**2)) / (DT**2)
        D = -2 * DX / DT          # simplified from -2*DX*DT^2/(DT^3)
        E = DX**2 / (DT**2)       # simplified from DX^2*DT^2/(DT^4)

        # Begin Ferrari's method
        alpha = -3 * B**2 / (8 * A**2) + C / A
        beta  = B**3 / (8 * A**3) - B * C / (2 * A**2) + D / A
        gamma = -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A

        roots = []
        if abs(beta) < self.TOL:
            # The quartic is bi-quadratic; solve directly.
            discriminant = alpha**2 - 4 * gamma
            # Protect against small negative due to rounding.
            if discriminant < 0:
                discriminant = 0
            sqrt_disc = cmath.sqrt(discriminant)
            # Two candidate square-root expressions.
            try:
                term1 = cmath.sqrt((-alpha + sqrt_disc) / 2)
            except ValueError:
                term1 = 0
            try:
                term2 = cmath.sqrt((-alpha - sqrt_disc) / 2)
            except ValueError:
                term2 = 0
            x1 = -B/(4*A) + term1
            x2 = -B/(4*A) - term1
            x3 = -B/(4*A) + term2
            x4 = -B/(4*A) - term2
            roots = [x1, x2, x3, x4]
        else:
            P = -alpha**2/12 - gamma
            Q = -alpha**3/108 + alpha * gamma / 3 - beta**2/8
            inner = Q**2/4 + P**3/27
            sqrt_inner = cmath.sqrt(inner) if inner >= 0 else 0
            Rm = Q/2 - sqrt_inner
            # Compute the real cube root, preserving sign.
            U = math.copysign(abs(Rm)**(1/3), Rm) if abs(Rm) > self.TOL else 0
            if abs(U) < self.TOL:
                y = -5/6 * alpha - U
            else:
                y = -5/6 * alpha - U + P/(3*U)
            temp = alpha + 2*y
            W = cmath.sqrt(temp) if temp >= 0 else 0

            def safe_sqrt(val):
                return cmath.sqrt(val)

            common = 3*alpha + 2*y
            try:
                r1 = -B/(4*A) + 0.5*(W + safe_sqrt(-common - 2*beta/W))
                r2 = -B/(4*A) + 0.5*(-W + safe_sqrt(-common + 2*beta/W))
                r3 = -B/(4*A) + 0.5*(W - safe_sqrt(-common - 2*beta/W))
                r4 = -B/(4*A) + 0.5*(-W - safe_sqrt(-common + 2*beta/W))
                roots = [r1, r2, r3, r4]
            except Exception:
                roots = []

        # Multiply candidate x by DT to get xi.
        candidate = None
        for x in roots:
            # In MATLAB, they check the “imaginary part” (scaled by DT).
            # Here, our computations used cmath.sqrt so x is a float.
            xr = x
            axi = 0  # no imaginary part expected
            xt = xr * DT
            if DX >= 0:
                if 0 <= xt <= DX and axi < self.TOL:
                    candidate = xt
                    break
            else:
                if DX <= xt <= 0 and axi < self.TOL:
                    candidate = xt
                    break

        if candidate is not None:
            return candidate
        else:
            from application.interface2_service import Interface2Service
            from domain.interface2 import Interface2Parameters
            params = Interface2Parameters(cr=self.cr, df=self.DF, dp=self.DT, dpf=self.DX)
            service = Interface2Service(params)
            
            # Define the function whose root we seek.
            def f(x):
                return service.compute(x)
            
            # Choose the interval [0, DX] (or [DX, 0] if DX is negative)
            a, b = (0, DX) if DX >= 0 else (DX, 0)
            # Use Brent’s method from SciPy.
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=self.TOL)
            return root * DT

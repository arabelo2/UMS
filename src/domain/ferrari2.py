# domain/ferrari2.py

import numpy as np
from scipy.optimize import root_scalar
from domain.interface2 import interface2

def ferrari2_scalar(cr, DF, DT, DX):
    """
    Solve for the intersection point, xi (in mm), using Ferrari's method for scalar inputs.

    Parameters:
        cr  : float
              Ratio c1/c2 (wave speed in medium one divided by wave speed in medium two).
        DF  : float
              Depth in medium two (DF, in mm). Must be positive.
        DT  : float
              Height in medium one (DT, in mm). Must be non-negative.
        DX  : float
              Separation distance along the interface (in mm); can be positive or negative.

    Returns:
        xi  : float
              The computed intersection point (in mm). If a valid candidate is found via Ferrari’s method,
              xi is computed as (candidate root)*DT. Otherwise, the fallback root finder returns xi directly.
    """
    tol = 1e-6

    # Input validation
    if not isinstance(cr, (int, float)) or cr <= 0:
        raise ValueError("cr must be a positive number.")
    if not isinstance(DF, (int, float)) or DF <= 0:
        raise ValueError("DF must be positive.")
    if not isinstance(DT, (int, float)):
        raise ValueError("DT must be a number.")
    
    # Handle DT < 0 by adjusting the problem setup
    if DT < 0:
        # If DT is negative, flip the problem to make DT positive
        DT = abs(DT)
        DX = -DX  # Flip DX to maintain the correct geometry

    # If the media are nearly identical, use the explicit solution.
    if abs(cr - 1) < tol:
        return DX * DT / (DF + DT)
    
    cri = 1 / cr  # cri = c2/c1
    
    # Define coefficients of the quartic: A*x^4 + B*x^3 + C*x^2 + D*x + E = 0
    A = 1 - cri**2
    B = (2 * cri**2 * DX - 2 * DX) / DT
    C = (DX**2 + DT**2 - cri**2 * (DX**2 + DF**2)) / (DT**2)
    D = -2 * DX * DT**2 / (DT**3)  # simplifies to -2*DX/DT
    E = DX**2 * DT**2 / (DT**4)     # simplifies to DX**2/(DT**2)
    
    # Begin Ferrari's solution.
    alpha = -3 * B**2 / (8 * A**2) + C / A
    beta  = B**3 / (8 * A**3) - B * C / (2 * A**2) + D / A
    gamma = -3 * B**4 / (256 * A**4) + C * B**2 / (16 * A**3) - B * D / (4 * A**2) + E / A
    
    x_candidates = np.zeros(4, dtype=complex)
    
    # Compare scalar beta with tol.
    if abs(beta) < tol:
        # Quartic reduces to a bi-quadratic.
        sqrt_term = np.sqrt(alpha**2 - 4 * gamma + 0j)
        x_candidates[0] = -B/(4*A) + np.sqrt((-alpha + sqrt_term)/2 + 0j)
        x_candidates[1] = -B/(4*A) + np.sqrt((-alpha - sqrt_term)/2 + 0j)
        x_candidates[2] = -B/(4*A) - np.sqrt((-alpha + sqrt_term)/2 + 0j)
        x_candidates[3] = -B/(4*A) - np.sqrt((-alpha - sqrt_term)/2 + 0j)
    else:
        P = -alpha**2/12 - gamma
        Q = -alpha**3/108 + alpha*gamma/3 - beta**2/8
        disc = Q**2/4 + P**3/27
        Rm = Q/2 - np.sqrt(disc + 0j)
        # Use np.power to compute cube root, which supports complex numbers.
        U = np.power(Rm, 1/3.0)
        if abs(U) < tol:
            y_val = -5/6 * alpha - U
        else:
            y_val = -5/6 * alpha - U + P/(3*U)
        W = np.sqrt(alpha + 2*y_val + 0j)
        x_candidates[0] = -B/(4*A) + 0.5 * ( W + np.sqrt( -(3*alpha + 2*y_val + 2*beta/W) + 0j))
        x_candidates[1] = -B/(4*A) + 0.5 * (-W + np.sqrt( -(3*alpha + 2*y_val - 2*beta/W) + 0j))
        x_candidates[2] = -B/(4*A) + 0.5 * ( W - np.sqrt( -(3*alpha + 2*y_val + 2*beta/W) + 0j))
        x_candidates[3] = -B/(4*A) + 0.5 * (-W - np.sqrt( -(3*alpha + 2*y_val - 2*beta/W) + 0j))
    
    flag = False
    xi = None
    # Candidate branch: each candidate x is unscaled; compute xi = x * DT.
    for candidate in x_candidates:
        xr = np.real(candidate)
        axi = DT * abs(np.imag(candidate))
        xt = xr * DT
        if DX >= 0:
            if (xt >= 0 and xt <= DX) and (axi < tol):
                xi = xr * DT
                flag = True
                break
        else:
            if (xt <= 0 and xt >= DX) and (axi < tol):
                xi = xr * DT
                flag = True
                break
    
    if not flag:
        # Fallback: solve for xi directly using interface2.
        def f_interface2(x_val):
            from domain.interface2 import interface2
            # x_val is in mm, since interface2 expects the final intersection point.
            return interface2(x_val, cr, DF, DT, DX)
        if DX >= 0:
            a, b_val = 0, DX
        else:
            a, b_val = DX, 0
        try:
            sol = root_scalar(f_interface2, bracket=[a, b_val], method='brentq', xtol=tol)
            if sol.converged:
                xi = sol.root
            else:
                xi = DX * DT / (DF + DT)  # Fallback explicit solution
        except ValueError:
            xi = DX * DT / (DF + DT)  # Handle errors in root finding

    return xi

def ferrari2(cr, DF, DT, DX):
    """
    Vectorized wrapper for ferrari2_scalar.
    If DF is not a scalar, apply ferrari2_scalar element-wise.
    """
    if np.isscalar(DF):
        return ferrari2_scalar(cr, DF, DT, DX)
    else:
        vectorized_func = np.vectorize(ferrari2_scalar)
        return vectorized_func(cr, DF, DT, DX)

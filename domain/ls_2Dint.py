# domain/ls_2Dint.py

import math
import numpy as np
from domain.pts_2Dintf import pts_2Dintf

def ls_2Dint(b, f, c, e, mat, angt, Dt0, x, z, Nopt=None):
    """
    Compute the normalized pressure p at a location (x, z) (in mm) in the second fluid
    for a source of length 2*b (mm) radiating waves across a fluid-fluid interface using
    a Rayleigh-Sommerfeld type integral with ray theory for cylindrical wave propagation.
    
    Parameters:
      b    : float
             Half-length of the element (mm).
      f    : float
             Frequency (MHz).
      c    : float
             Wave speed in the fluid (m/s). (Not used directly; material speeds from mat are used.)
      e    : float
             Offset (mm) of the element from the array center.
      mat  : list or array of 4 elements [d1, c1, d2, c2], where:
             - d1: density in medium one (gm/cm³)
             - c1: wave speed in medium one (m/s)
             - d2: density in medium two (gm/cm³)
             - c2: wave speed in medium two (m/s)
      angt : float
             Angle (degrees) that the array makes with the x-axis.
      Dt0  : float
             Distance (mm) from the array center to the interface.
      x    : float or numpy array
             x-coordinate (mm) in the second fluid where pressure is computed.
      z    : float or numpy array
             z-coordinate (mm) in the second fluid where pressure is computed.
      Nopt : int, optional
             Optional number of segments to use in the numerical integration.
             If not provided, N is computed as round(2000 * f * b / c1) with a minimum of 1.
    
    Returns:
      p : complex or numpy array of complex numbers
          The normalized pressure.
    
    Procedure:
      1. Extract material parameters from mat.
      2. Compute wave numbers:
            k1b = 2000 * pi * b * f / c1
            k2b = 2000 * pi * b * f / c2
      3. Determine the number of segments, N.
      4. Compute centroid positions for each segment:
            For jj = 1,...,N: xc(jj) = b * (-1 + 2*(jj - 0.5)/N)
      5. For each segment:
            a. Call pts_2Dintf(e, xc[jj], angt, Dt0, c1, c2, x, z) to compute xi_seg.
            b. Compute:
                 Dtn = Dt0 + (e + xc(jj)) * sin(angt in radians)
                 Dxn = x - (e + xc(jj)) * cos(angt in radians)
            c. Compute:
                 r1 = sqrt(xi_seg**2 + Dtn**2) / b
                 r2 = sqrt((Dxn - xi_seg)**2 + z**2) / b
            d. Compute angles:
                 ang1 = arcsin(xi_seg / (b * r1))   (with protection for division by zero)
                 ang2 = arcsin((Dxn - xi_seg) / (b * r2))
                 ang  = (angt in radians) - ang1, and if sin(ang)==0 add a tiny epsilon.
            e. Compute directivity:
                 dir = sin(k1b*sin(ang)/N) / (k1b*sin(ang)/N)
            f. Compute transmission coefficient:
                 Tp = 2*d2*c2*cos(ang1) / (d1*c1*cos(ang2) + d2*c2*cos(ang1))
            g. Compute phase term:
                 ph = exp(1j * k1b*r1 + 1j * k2b*r2)
            h. Compute denominator:
                 den = r1 + (c2/c1)*r2*(np.cos(ang1)**2)/(np.cos(ang2)**2)
            i. Accumulate:
                 p = p + Tp * dir * ph / sqrt(den)
      6. Multiply p by the external factor: p = p * (sqrt(2*k1b/(1j*pi)))/N.
    """
    d1, c1, d2, c2 = mat
    # Compute wave numbers.
    k1b = 2000 * math.pi * b * f / c1
    k2b = 2000 * math.pi * b * f / c2
    
    # Determine number of segments.
    if Nopt is not None:
        N = Nopt
    else:
        N = round(2000 * f * b / c1)
        if N < 1:
            N = 1
    
    # Compute centroid positions.
    xc = np.zeros(N, dtype=float)
    for jj in range(1, N+1):
        xc[jj-1] = b * (-1 + 2*(jj - 0.5)/N)
    
    p = 0  # Initialize the pressure accumulator.
    eps_val = np.finfo(float).eps  # Machine epsilon.
    angt_rad = np.deg2rad(angt)
    
    # Loop over segments.
    for xn in xc:
        # Compute intersection point for this segment.
        xi_seg = pts_2Dintf(e, xn, angt, Dt0, c1, c2, x, z)
        # Compute effective distances.
        Dtn = Dt0 + (e + xn) * np.sin(angt_rad)
        Dxn = x - (e + xn) * np.cos(angt_rad)
        r1 = np.sqrt(xi_seg**2 + Dtn**2) / b
        r2 = np.sqrt((Dxn - xi_seg)**2 + z**2) / b
        # Compute angles in a vectorized, safe way.
        ang1 = np.where(r1 == 0, 0, np.arcsin(xi_seg / (b * r1)))
        ang2 = np.where(r2 == 0, 0, np.arcsin((Dxn - xi_seg) / (b * r2)))
        ang = angt_rad - ang1
        # Avoid division by zero in the directivity calculation.
        ang = np.where(np.abs(np.sin(ang)) < eps_val, ang + eps_val, ang)
        dir_factor = np.sin(k1b * np.sin(ang) / N) / (k1b * np.sin(ang) / N)
        Tp = 2 * d2 * c2 * np.cos(ang1) / (d1 * c1 * np.cos(ang2) + d2 * c2 * np.cos(ang1))
        ph = np.exp(1j * k1b * r1 + 1j * k2b * r2)
        den = r1 + (c2 / c1) * r2 * (np.cos(ang1) ** 2) / (np.cos(ang2) ** 2)
        p += Tp * dir_factor * ph / np.sqrt(den)
    
    p = p * (np.sqrt(2 * k1b / (1j * math.pi))) / N
    return p

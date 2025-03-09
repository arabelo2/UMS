#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def ps_3Dint(lx, ly, f, mat, ex, ey, angt, Dt0, x, y, z):
    """
    Python implementation of the MATLAB ps_3Dint function.
    Computes the normalized velocity components (vx, vy, vz) for a rectangular array element.
    """
    # Extract material properties
    d1, cp1, d2, cp2, cs2, wave_type = mat

    # Wave speeds
    c1 = cp1
    c2 = cp2 if wave_type == 'p' else cs2

    # Wave numbers
    k1 = 2000 * np.pi * f / c1
    k2 = 2000 * np.pi * f / c2

    # Number of segments (default: one segment per wavelength)
    R = max(1, int(np.ceil(1000 * f * lx / c1)))
    Q = max(1, int(np.ceil(1000 * f * ly / c1)))

    # Segment centroids
    xc = np.linspace(-lx/2 + lx/(2*R), lx/2 - lx/(2*R), R)
    yc = np.linspace(-ly/2 + ly/(2*Q), ly/2 - ly/(2*Q), Q)

    # Initialize velocity components
    vx = np.zeros_like(x, dtype=complex)
    vy = np.zeros_like(x, dtype=complex)
    vz = np.zeros_like(x, dtype=complex)

    # Segment sizes
    dx = lx / R
    dy = ly / Q

    # Loop over segments
    for rr in range(R):
        for qq in range(Q):
            # Distance along the interface
            Db = np.sqrt((x - (ex + xc[rr]) * np.cos(np.deg2rad(angt)))**2 + (y - (ey + yc[qq]))**2)
            Ds = Dt0 + (ex + xc[rr]) * np.sin(np.deg2rad(angt))
            xi = pts_3Dint(ex, ey, xc[rr], yc[qq], angt, Dt0, c1, c2, x, y, z)

            # Incident and refracted angles
            ang1 = np.where(Db != 0, np.arctan2(xi, Ds), 0)
            ang2 = np.where(ang1 != 0, np.arctan2(Db - xi, z), 0)

            # Ray path lengths
            r1 = np.sqrt(Ds**2 + xi**2)
            r2 = np.sqrt((Db - xi)**2 + z**2)

            # Polarization components
            px = np.where(Db != 0, (1 - xi/Db) * (x - (ex + xc[rr]) * np.cos(np.deg2rad(angt))) / r2, 0)
            py = np.where(Db != 0, (1 - xi/Db) * (y - (ey + yc[qq])) / r2, 0)
            pz = np.where(Db != 0, z / r2, 1)

            # Transmission coefficients
            tpp, tps = T_fluid_solid(d1, cp1, d2, cp2, cs2, np.rad2deg(ang1))
            T = np.where(wave_type == 'p', tpp, tps)

            # Directivity term
            argx = k1 * (x - (ex + xc[rr]) * np.cos(np.deg2rad(angt))) * dx / (2 * r1)
            argy = k1 * (y - (ey + yc[qq])) * dy / (2 * r1)
            dir_term = (np.sinc(argx / np.pi) * np.sinc(argy / np.pi))

            # Denominator terms
            D1 = r1 + r2 * (c2 / c1) * (np.cos(ang1) / np.cos(ang2))**2
            D2 = r1 + r2 * (c2 / c1)

            # Velocity components
            phase_term = np.exp(1j * k1 * r1 + 1j * k2 * r2)
            vx += T * px * dir_term * phase_term / np.sqrt(D1 * D2)
            vy += T * py * dir_term * phase_term / np.sqrt(D1 * D2)
            vz += T * pz * dir_term * phase_term / np.sqrt(D1 * D2)

    # External factor
    factor = (-1j * k1 * dx * dy) / (2 * np.pi)
    vx *= factor
    vy *= factor
    vz *= factor

    return vx, vy, vz


def pts_3Dint(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    """
    Python implementation of the MATLAB pts_3Dint function.
    Computes the intersection point xi on the interface.
    """
    cr = c1 / c2
    De = Dt0 + (ex + xn) * np.sin(np.deg2rad(angt))
    Db = np.sqrt((x - (ex + xn) * np.cos(np.deg2rad(angt)))**2 + (y - (ey + yn))**2)
    
    # Initialize xi array with the same shape as z
    xi = np.zeros_like(z)
    
    # Iterate over each element in z and Db
    for i in range(z.size):
        xi.flat[i] = ferrari2(cr, z.flat[i], De, Db.flat[i])
    return xi


def ferrari2(cr, DF, DT, DX):
    """
    Python implementation of the MATLAB ferrari2 function.
    Solves for the intersection point xi using a bounded solver.
    """
    if np.abs(cr - 1) < 1e-6:
        return DX * DT / (DF + DT)
    else:
        # Use a bounded solver to find the root of interface2
        try:
            result = root_scalar(
                lambda xi: interface2(xi, cr, DF, DT, DX),
                bracket=[0, DX],  # Search within [0, DX]
                method='brentq'   # Brent's method for robustness
            )
            return result.root
        except ValueError:
            # Fallback to a linear approximation if the solver fails
            return DX * DT / (DF + DT)


def interface2(x, cr, df, dp, dpf):
    """
    Python implementation of the MATLAB interface2 function.
    Evaluates the Snell's law condition for a candidate intersection point x.
    """
    return x / np.sqrt(x**2 + dp**2) - cr * (dpf - x) / np.sqrt((dpf - x)**2 + df**2)


def T_fluid_solid(d1, cp1, d2, cp2, cs2, theta1):
    """
    Python implementation of the MATLAB T_fluid_solid function.
    Computes the transmission coefficients tpp and tps.
    """
    # Placeholder implementation (replace with actual calculation)
    tpp = 1.0
    tps = 0.0
    return tpp, tps


# Parameters
lx = 6
ly = 12
f = 5
mat = [1, 1480, 7.9, 5900, 3200, 'p']
angt = 10.217
Dt0 = 50.8
x = np.linspace(0, 30, 100)
z = np.linspace(1, 20, 100)
y = 0
xx, zz = np.meshgrid(x, z)

# Compute velocity components
vx, vy, vz = ps_3Dint(lx, ly, f, mat, 0, 0, angt, Dt0, xx, y, zz)

# Compute magnitude of velocity
v = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)

# Plot the result
plt.imshow(v,
           cmap="jet", extent=[x.min(), x.max(), z.min(), z.max()],
           aspect='auto')
plt.colorbar(label='Velocity Magnitude')
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.title('Velocity Magnitude in the Second Medium')
plt.show()
# domain/ls_2Dint.py

"""
Module: ls_2Dint.py
Layer: Domain

This module implements the LS2DInterface class that computes the normalized
pressure for a 1-D array element radiating waves across a plane fluid/fluid interface,
using a Rayleigh-Sommerfeld type model.

Note: This version is fully vectorized using NumPy so that x and z can be 2-D arrays
(e.g. a meshgrid). It depends on the Pts2DIntfService from the application layer.
"""

import numpy as np
from application.pts_2Dintf_service import Pts2DIntfService

class LS2DInterface:
    """
    Domain class for computing the normalized pressure p for an element in a 1-D array
    radiating waves across a plane fluid/fluid interface.

    Parameters:
      b    : half-length of the source (mm)
      f    : frequency (MHz)
      mat  : material properties [d1, c1, d2, c2] where d1 and c1 are the density (gm/cm^3)
             and wave speed (m/s) in medium 1, and d2 and c2 are the density and wave speed
             in medium 2.
      e    : offset of the element from the array center (mm)
      angt : angle (in degrees) of the array with respect to the x-axis
      Dt0  : distance of the array center above the interface (mm)
      x, z : coordinates (mm) at which pressure is computed in medium 2; these may be scalars or arrays.
      Nopt : (optional) number of segments to use; if not provided, it is computed.
    """
    def __init__(self, b: float, f: float, mat: list, e: float, angt: float, Dt0: float, x, z, Nopt: int = None):
        self.b = b
        self.f = f
        if isinstance(mat, (list, tuple)) and len(mat) == 4:
            self.d1, self.c1, self.d2, self.c2 = mat
        else:
            raise ValueError("mat must be a list or tuple of four elements: [d1, c1, d2, c2]")
        self.e = e
        self.angt = angt
        self.Dt0 = Dt0
        self.x = x
        self.z = z
        
        # Compute wave numbers.
        self.k1b = 2000 * np.pi * b * f / self.c1
        self.k2b = 2000 * np.pi * b * f / self.c2
        
        # Determine the number of segments.
        if Nopt is not None:
            self.N = Nopt
        else:
            self.N = round(2000 * f * b / self.c1)
            if self.N < 1:
                self.N = 1

        # Compute centroid locations for the segments.
        self.xc = np.array([b * (-1 + 2 * (jj - 0.5) / self.N) for jj in range(1, self.N + 1)])

    def compute(self) -> np.complex128:
        
        # Convert self.x to a NumPy array.
        x_arr = np.array(self.x)
        # If x_arr is a scalar, we set p to a scalar; if it's an array, initialize p with the same shape.
        if x_arr.ndim == 0:
            p = 0.0 + 0.0j
        else:
            p = np.zeros(x_arr.shape, dtype=np.complex128)

        # Loop over each segment defined by its offset xn.
        # For each segment, compute the contribution from that segment.
        for xn in self.xc:                        
            pts_service = Pts2DIntfService(self.e, xn, self.angt, self.Dt0, self.c1, self.c2, self.x, self.z)
            xi = pts_service.compute()  # xi may be a scalar or an array matching the shape of x.
            
            # Compute effective vertical distance (Dtn) as a scalar.
            Dtn = self.Dt0 + (self.e + xn) * np.sin(np.radians(self.angt))
            # Compute horizontal distance (Dxn) elementwise.
            Dxn = np.array(self.x) - (self.e + xn) * np.cos(np.radians(self.angt))
            
            # Compute normalized distances.
            r1 = np.sqrt(xi**2 + Dtn**2) / self.b
            r2 = np.sqrt((Dxn - xi)**2 + np.array(self.z)**2) / self.b
            
            # Compute angles, using np.clip to ensure valid domain for arcsin.
            ang1 = np.arcsin(xi / (self.b * r1))
            ang2 = np.arcsin((Dxn - xi) / (self.b * r2))
            ang = np.radians(self.angt) - ang1
            
            ang = ang + np.finfo(float).eps * (ang == 0)
            
            # Compute the segment directivity.
            dir_factor = np.sinc(self.k1b * np.sin(ang) / (np.pi * self.N))
            
            # Compute the plane wave transmission coefficient.
            Tp = (2 * self.d2 * self.c2 * np.cos(ang1)) / (
                self.d1 * self.c1 * np.cos(ang2) + self.d2 * self.c2 * np.cos(ang1)
            )
            
            # Compute phase term.
            phase = np.exp(1j * self.k1b * r1 + 1j * self.k2b * r2)
            
            # Compute denominator.
            denominator = r1 + (self.c2 / self.c1) * r2 * (np.cos(ang1)**2) / (np.cos(ang2)**2)
            
            # Compute the segment contribution elementwise.
            seg_contrib = Tp * dir_factor * phase / np.sqrt(denominator)
            
            # Accumulate contribution elementwise.
            p = p + seg_contrib

        # External factor – compute the square root of a complex number using np.sqrt.
        external_factor = np.sqrt(2 * self.k1b / (1j * np.pi)) / self.N
        p = p * external_factor
        
        return p

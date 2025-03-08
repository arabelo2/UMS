# domain/ps_3Dv.py

import math
import numpy as np

class RectangularPiston3D:
    def __init__(self, lx: float, ly: float, f: float, c: float, ex: float, ey: float):
        """
        Initialize the rectangular piston for 3D pressure computation.

        Parameters:
            lx (float): Element length along the x-axis (mm).
            ly (float): Element length along the y-axis (mm).
            f (float): Frequency in MHz.
            c (float): Wave speed in m/s.
            ex (float): Lateral offset along the x-axis (mm).
            ey (float): Lateral offset along the y-axis (mm).
        """
        self.lx = lx
        self.ly = ly
        self.f = f
        self.c = c
        self.ex = ex
        self.ey = ey
        # Compute wave number: k = 2000*pi*f/c (with f in MHz, c in m/s)
        self.k = 2000 * math.pi * self.f / self.c

    def compute_pressure(self, x, y, z, P: int = None, Q: int = None):
        """
        Compute the normalized pressure at evaluation point(s) (x, y, z) using the 
        Rayleigh–Sommerfeld integral for a rectangular piston source.

        Parameters:
            x, y, z: Scalars or NumPy arrays (in mm) where the pressure is evaluated.
            P (int, optional): Number of segments in the x-direction. If not provided,
                               it is computed as ceil(1000*f*lx/c) with a minimum of 1.
            Q (int, optional): Number of segments in the y-direction. If not provided,
                               it is computed as ceil(1000*f*ly/c) with a minimum of 1.

        Returns:
            p: The computed normalized pressure (complex scalar or NumPy array) 
               matching the broadcast shape of the inputs.
        """
        # Determine the number of segments in the x-direction.
        if P is None:
            P = math.ceil(1000 * self.f * self.lx / self.c)
            if P < 1:
                P = 1
        # Determine the number of segments in the y-direction.
        if Q is None:
            Q = math.ceil(1000 * self.f * self.ly / self.c)
            if Q < 1:
                Q = 1

        # Compute centroids for segments in x and y.
        xc = np.array([ -self.lx/2 + (self.lx / P) * (pp - 0.5) for pp in range(1, P+1) ])
        yc = np.array([ -self.ly/2 + (self.ly / Q) * (qq - 0.5) for qq in range(1, Q+1) ])

        # Convert evaluation points to NumPy arrays.
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        z_arr = np.array(z, dtype=float)
        # Broadcast shape for x, y, z.
        broadcast_shape = np.broadcast(x_arr, y_arr, z_arr).shape
        p = np.zeros(broadcast_shape, dtype=complex)

        # Small epsilon to avoid division by zero.
        eps = np.finfo(float).eps

        # Loop over segments.
        for xi in xc:
            for yi in yc:
                # Compute distance from the sub-element centroid to evaluation point.
                rpq = np.sqrt((x_arr - xi - self.ex)**2 + (y_arr - yi - self.ey)**2 + z_arr**2)
                rpq = np.where(rpq == 0, eps, rpq)
                # Compute direction cosines.
                ux = (x_arr - xi - self.ex) / rpq
                uy = (y_arr - yi - self.ey) / rpq
                ux = np.where(ux == 0, ux + eps, ux)
                uy = np.where(uy == 0, uy + eps, uy)
                # Compute directivity factors.
                argx = self.k * ux * self.lx / (2 * P)
                argy = self.k * uy * self.ly / (2 * Q)
                # Avoid division by zero in sinc calculations.
                dirx = np.where(np.abs(argx) < eps, 1.0, np.sin(argx) / argx)
                diry = np.where(np.abs(argy) < eps, 1.0, np.sin(argy) / argy)
                # Accumulate contribution of this sub-element.
                p += dirx * diry * np.exp(1j * self.k * rpq) / rpq

        # Multiply by the external factor.
        factor = (-1j * self.k * (self.lx / P) * (self.ly / Q)) / (2 * math.pi)
        p = p * factor
        return p

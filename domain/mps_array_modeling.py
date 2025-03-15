# domain/mps_array_modeling.py

import numpy as np
import math
from application.delay_laws3D_service import run_delay_laws3D_service
from application.ps_3Dv_service import run_ps_3Dv_service
from application.discrete_windows_service import run_discrete_windows_service

class MPSArrayModeling:
    """
    Class to compute the normalized pressure field of a 2D array of rectangular elements.

    Parameters:
      lx, ly : float
          Element dimensions in x and y (mm).
      gx, gy : float
          Gap lengths in x and y (mm).
      f : float
          Frequency (MHz).
      c : float
          Wave speed (m/s).
      L1, L2 : int
          Number of elements in x- and y-directions.
      theta, phi : float
          Steering angles (degrees).
      Fl : float
          Focal distance (mm); use np.inf for steering-only.
      ampx_type, ampy_type : str
          Window types for apodization in x and y directions.
      xs, zs : np.ndarray, optional
          1D arrays defining the evaluation grid in x and z (mm). Defaults: xs from -15 to 15 (300 pts), zs from 1 to 20 (200 pts).
      y : float, optional
          Fixed y-coordinate for evaluation (default=0).
    """
    def __init__(self, lx: float, ly: float, gx: float, gy: float,
                 f: float, c: float, L1: int, L2: int,
                 theta: float, phi: float, Fl: float,
                 ampx_type: str, ampy_type: str,
                 xs: np.ndarray = None, zs: np.ndarray = None, y: float = 0.0):
        self.lx = lx
        self.ly = ly
        self.gx = gx
        self.gy = gy
        self.f = f
        self.c = c
        self.L1 = L1
        self.L2 = L2
        self.theta = theta
        self.phi = phi
        self.Fl = Fl
        self.ampx_type = ampx_type
        self.ampy_type = ampy_type

        # Total pitches (element + gap)
        self.sx_total = lx + gx
        self.sy_total = ly + gy

        # Set evaluation grid defaults if not provided
        if xs is None:
            self.xs = np.linspace(-15, 15, 300)
        else:
            self.xs = xs
        if zs is None:
            self.zs = np.linspace(1, 20, 200)
        else:
            self.zs = zs
        self.y = y

        # Create evaluation grid (x, z) with constant y
        self.x, self.z = np.meshgrid(self.xs, self.zs)
        self.y_arr = np.full(self.x.shape, y)
    
    def compute_pressure_field(self):
        """
        Compute the normalized pressure field.

        Returns:
           p : np.ndarray (complex)
              Normalized pressure field evaluated over the grid.
           xs : np.ndarray
              x coordinates used in the grid.
           zs : np.ndarray
              z coordinates used in the grid.
        """
        # Compute element centroid positions in x and y directions.
        Nx = np.arange(1, self.L1 + 1)
        Mb = (self.L1 - 1) / 2.0
        ex = (2 * Nx - 1 - self.L1) * (self.sx_total / 2)

        Ny = np.arange(1, self.L2 + 1)
        Nb = (self.L2 - 1) / 2.0
        ey = (2 * Ny - 1 - self.L2) * (self.sy_total / 2)

        # Compute time delays (in microseconds) for the array.
        td = run_delay_laws3D_service(self.L1, self.L2, self.sx_total, self.sy_total,
                                      self.theta, self.phi, self.Fl, self.c)
        # Compute the delay phase factor:
        delay_phase = np.exp(1j * 2 * np.pi * self.f * td)

        # Compute amplitude weights using discrete windows.
        Cx = run_discrete_windows_service(self.L1, self.ampx_type)
        Cy = run_discrete_windows_service(self.L2, self.ampy_type)

        # Initialize the pressure field.
        p = np.zeros(self.x.shape, dtype=complex)

        # Loop over each array element and accumulate pressure contributions.
        for i in range(self.L1):
            for j in range(self.L2):
                weight = Cx[i] * Cy[j]
                d_factor = delay_phase[i, j]
                # Compute pressure from the rectangular piston for this element.
                p_elem = run_ps_3Dv_service(self.lx, self.ly, self.f, self.c,
                                            ex[i], ey[j], self.x, self.y_arr, self.z)
                p += weight * d_factor * p_elem

        return p, self.xs, self.zs

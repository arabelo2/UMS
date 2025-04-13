# domain/mps_array_model_int.py

import numpy as np
import math
from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from application.discrete_windows_service import run_discrete_windows_service
from application.ps_3Dint_service import run_ps_3Dint_service

class MPSArrayModelInt:
    """
    Class to compute the normalized velocity wave field for an array of 1-D elements
    radiating waves through a fluid/solid interface.
    """
    def __init__(self, lx: float, ly: float, gx: float, gy: float,
                 f: float, d1: float, cp1: float, d2: float, cp2: float, cs2: float, wave_type: str,
                 L1: int, L2: int, angt: float, Dt0: float,
                 theta20: float, phi: float, DF: float,
                 ampx_type: str, ampy_type: str,
                 xs, zs, y):
        """
        Initialize the MPS Array Modeling parameters.
        """
        # Validate inputs (existing checks remain)
        if lx <= 0:
            raise ValueError("lx must be positive.")
        if ly <= 0:
            raise ValueError("ly must be positive.")
        if f <= 0:
            raise ValueError("Frequency f must be positive.")
        if Dt0 < 0:
            raise ValueError("Dt0 must be non-negative.")
        if L1 <= 0 or L2 <= 0:
            raise ValueError("L1 and L2 must be positive integers.")
        if gx < 0 or gy < 0:
            raise ValueError("gx and gy must be non-negative.")
        if xs is None or len(xs) == 0:
            raise ValueError("xs must be provided and non-empty.")
        if zs is None or len(zs) == 0:
            raise ValueError("zs must be provided and non-empty.")

        self.lx = lx
        self.ly = ly
        self.gx = gx
        self.gy = gy
        self.f = f
        self.d1 = d1
        self.cp1 = cp1
        self.d2 = d2
        self.cp2 = cp2
        self.cs2 = cs2
        self.wave_type = wave_type.lower()
        # Construct material vector
        self.mat = [d1, cp1, d2, cp2, cs2, self.wave_type]
        
        self.L1 = L1
        self.L2 = L2
        self.angt = angt
        self.Dt0 = Dt0
        self.theta20 = theta20
        self.phi = phi
        self.DF = DF
        self.ampx_type = ampx_type
        self.ampy_type = ampy_type
        
        self.xs = xs
        self.zs = zs
        self.y = y

    def compute_field(self):
        """
        Compute the normalized velocity field.

        Returns:
            dict: Contains:
                'p' - Normalized velocity magnitude (2D or 3D np.ndarray).
                'x' - x-coordinates used.
                'z' - z-coordinates used.
                Note: When evaluation is 3D, the grid shape is (len(xs), len(y), len(z)).
        """
        # Convert inputs to numpy arrays.
        x_vals = np.array(self.xs, dtype=float)
        y_vals = np.array(self.y, dtype=float)
        z_vals = np.array(self.zs, dtype=float)
        
        # Helper to determine if a value is a vector.
        def is_vector(val):
            arr = np.atleast_1d(val)
            return arr.size > 1
        
        x_is_vec = is_vector(x_vals)
        y_is_vec = is_vector(y_vals)
        z_is_vec = is_vector(z_vals)
        vec_count = sum([x_is_vec, y_is_vec, z_is_vec])
        
        if vec_count <= 1:
            mode = "1D"
        elif vec_count == 2:
            mode = "2D"
        elif vec_count == 3:
            mode = "3D"
        else:
            raise ValueError("Unexpected coordinate combination.")
        
        # Build evaluation grid based on the mode.
        if mode == "1D":
            # All inputs are scalars or only one is an array.
            X_input = x_vals
            Y_input = y_vals
            Z_input = z_vals
        elif mode == "2D":
            if not x_is_vec:
                # x is scalar; create grid from y and z.
                Y_grid, Z_grid = np.meshgrid(np.atleast_1d(y_vals), np.atleast_1d(z_vals))
                X_grid = np.full(Y_grid.shape, x_vals)
                X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
            elif not y_is_vec:
                # y is scalar; create grid from x and z.
                X_grid, Z_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(z_vals))
                Y_grid = np.full(X_grid.shape, y_vals)
                X_input, Y_input, Z_input = X_grid, Y_grid, Z_grid
            elif not z_is_vec:
                # z is scalar; create grid from x and y.
                X_grid, Y_grid = np.meshgrid(np.atleast_1d(x_vals), np.atleast_1d(y_vals))
                Z_input = z_vals
                X_input, Y_input = X_grid, Y_grid
            else:
                # Fallback: if two are vectors.
                X_input = x_vals
                Y_input = y_vals
                Z_input = z_vals
        else:  # mode == "3D"
            X_input, Y_input, Z_input = np.meshgrid(np.atleast_1d(x_vals),
                                                     np.atleast_1d(y_vals),
                                                     np.atleast_1d(z_vals),
                                                     indexing='ij')
        
        # The rest of the computation remains as before.
        # Compute total pitches.
        sx_total = self.lx + self.gx
        sy_total = self.ly + self.gy

        # Compute element centroid locations.
        Nx = np.arange(1, self.L1 + 1)
        Mb = (self.L1 - 1) / 2.0
        ex = (Nx - 1 - Mb) * sx_total  # shape (L1,)
        
        Ny = np.arange(1, self.L2 + 1)
        Mby = (self.L2 - 1) / 2.0
        ey = (Ny - 1 - Mby) * sy_total  # shape (L2,)

        # Compute time delays using the delay_laws3Dint service.
        td, _ = run_delay_laws3Dint_service(
            Mx=self.L1, My=self.L2, sx=sx_total, sy=sy_total,
            theta=self.angt, phi=self.phi, theta20=self.theta20,
            DT0=self.Dt0, DF=self.DF, c1=self.cp1, c2=(self.cp2 if self.wave_type=='p' else self.cs2),
            plt_option='n', view_elev=25, view_azim=20, z_scale=1.0
        )
        delay_phase = np.exp(1j * 2 * np.pi * self.f * td)

        # Compute amplitude weights using discrete windows service.
        Cx = run_discrete_windows_service(self.L1, self.ampx_type)
        Cy = run_discrete_windows_service(self.L2, self.ampy_type)

        # Initialize velocity field components.
        vx = 0
        vy = 0
        vz = 0

        # Loop over each element to compute contribution from ps_3Dint.
        for nn in range(self.L1):
            for ll in range(self.L2):
                vxe, vye, vze = run_ps_3Dint_service(
                    self.lx, self.ly, self.f, self.mat,
                    ex[nn], ey[ll], self.angt, self.Dt0,
                    X_input, Y_input, Z_input
                )
                vx += Cx[nn] * Cy[ll] * delay_phase[nn, ll] * vxe
                vy += Cx[nn] * Cy[ll] * delay_phase[nn, ll] * vye
                vz += Cx[nn] * Cy[ll] * delay_phase[nn, ll] * vze

        # Compute velocity magnitude.
        vmag = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)
        # Return keys 'p', 'x', and 'z' for consistency with the interface layer.
        return {'p': vmag, 'x': x_vals, 'z': z_vals}

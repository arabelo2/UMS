# domain/ps_3Dint.py

import numpy as np
from domain.pts_3Dintf import Pts3DIntf
from domain.T_fluid_solid import FluidSolidTransmission

class Ps3DInt:
    """
    Core domain logic for computing velocity components (vx, vy, vz) using the ps_3Dint algorithm.
    """

    def __init__(self, lx, ly, f, mat, ex, ey, angt, Dt0):
        """
        Initialize the Ps3DInt object.

        Parameters:
            lx   : float - Length of the array element in the x-direction (mm).
            ly   : float - Length of the array element in the y-direction (mm).
            f    : float - Frequency of the wave (MHz).
            mat  : list  - Material properties [d1, cp1, d2, cp2, cs2, wave_type].
            ex   : float - Offset of the element center from the array center in x (mm).
            ey   : float - Offset of the element center from the array center in y (mm).
            angt : float - Angle of the array relative to the interface (degrees).
            Dt0  : float - Distance from the array center to the interface (mm).

        Raises:
            ValueError: If any input parameter is invalid.
        """
        # Validate inputs
        if lx <= 0:
            raise ValueError("lx must be a positive value.")
        if ly <= 0:
            raise ValueError("ly must be a positive value.")
        if f <= 0:
            raise ValueError("f must be a positive value.")
        if Dt0 < 0:  # Allow Dt0 == 0 now
            raise ValueError("Dt0 must be non-negative.")
        if not isinstance(mat, list) or len(mat) != 6:
            raise ValueError("mat must be a list of 6 elements.")
        if mat[-1] not in ['p', 's']:
            raise ValueError("mat must specify wave type as 'p' or 's'.")

        # Assign parameters
        self.lx = lx
        self.ly = ly
        self.f = f
        self.mat = mat
        self.ex = ex
        self.ey = ey
        self.angt = np.deg2rad(angt)  # Convert to radians
        self.Dt0 = Dt0

        # Extract material properties
        self.d1, self.cp1, self.d2, self.cp2, self.cs2, self.wave_type = mat

        # Wave speeds
        self.c1 = self.cp1
        self.c2 = self.cp2 if self.wave_type == 'p' else self.cs2

        # Wave numbers
        self.k1 = 2000 * np.pi * self.f / self.c1
        self.k2 = 2000 * np.pi * self.f / self.c2

    def compute_velocity_components(self, x, y, z):
        """
        Compute the velocity components (vx, vy, vz) for the given coordinates (x, y, z).

        Parameters:
            x : numpy array - x-coordinates (mm).
            y : numpy array - y-coordinates (mm).
            z : numpy array - z-coordinates (mm).

        Returns:
            tuple: (vx, vy, vz) - Velocity components as complex numpy arrays.
        """
        # Number of segments (default: one segment per wavelength)
        R = max(1, int(np.ceil(1000 * self.f * self.lx / self.c1)))
        Q = max(1, int(np.ceil(1000 * self.f * self.ly / self.c1)))

        # Segment centroids
        xc = np.linspace(-self.lx / 2 + self.lx / (2 * R), self.lx / 2 - self.lx / (2 * R), R)
        yc = np.linspace(-self.ly / 2 + self.ly / (2 * Q), self.ly / 2 - self.ly / (2 * Q), Q)

        # Initialize velocity components
        vx = np.zeros_like(x, dtype=complex)
        vy = np.zeros_like(x, dtype=complex)
        vz = np.zeros_like(x, dtype=complex)

        # Segment sizes
        dx = self.lx / R
        dy = self.ly / Q

        # Small epsilon to avoid division by zero
        eps = 1e-10

        # Loop over segments
        for rr in range(R):
            for qq in range(Q):
                # Compute Ds and clamp it to a small positive value if negative
                Ds = self.Dt0 + (self.ex + xc[rr]) * np.sin(self.angt)
                Ds = max(Ds, eps)  # Ensure Ds is non-negative

                # Compute intersection point xi
                pts = Pts3DIntf(self.ex, self.ey, xc[rr], yc[qq], np.rad2deg(self.angt), Ds, self.c1, self.c2)
                xi = pts.compute_intersection(x, y, z)

                # Compute incident and refracted angles
                Db = np.sqrt((x - (self.ex + xc[rr]) * np.cos(self.angt)) ** 2 +
                             (y - (self.ey + yc[qq])) ** 2)
                # Compute raw angle (radians) using arctan2
                raw_ang1 = np.where(Db != 0, np.arctan2(xi, Ds), 0)
                # Convert to degrees, take absolute value, and clip to [0, 90]
                ang1_deg = np.clip(np.abs(np.rad2deg(raw_ang1)), 0, 90)
                # Convert back to radians for further computations
                ang1 = np.deg2rad(ang1_deg)
                ang2 = np.where(ang1 != 0, np.arctan2(Db - xi, z), 0)

                # Ray path lengths
                r1 = np.sqrt(Ds**2 + xi**2)
                r2 = np.sqrt((Db - xi)**2 + z**2)

                # Avoid division by zero
                Db_safe = np.where(Db == 0, eps, Db)
                r2_safe = np.where(r2 == 0, eps, r2)

                # Polarization components
                px = np.where(Db != 0, (1 - xi / Db_safe) * (x - (self.ex + xc[rr]) * np.cos(self.angt)) / r2_safe, 0)
                py = np.where(Db != 0, (1 - xi / Db_safe) * (y - (self.ey + yc[qq])) / r2_safe, 0)
                pz = np.where(Db != 0, z / r2_safe, 1)

                # Transmission coefficients
                tpp, tps = FluidSolidTransmission.compute_coefficients(
                    self.d1, self.cp1, self.d2, self.cp2, self.cs2, np.rad2deg(ang1)
                )
                T = np.where(self.wave_type == 'p', tpp, tps)

                # Directivity term
                r1_safe = np.where(np.abs(r1) < eps, eps, r1)
                argx = self.k1 * (x - (self.ex + xc[rr]) * np.cos(self.angt)) * dx / (2 * r1_safe)
                argy = self.k1 * (y - (self.ey + yc[qq])) * dy / (2 * r1_safe)
                dir_term = (np.sinc(argx / np.pi) * np.sinc(argy / np.pi))

                # Denominator terms
                D1 = r1 + r2 * (self.c2 / self.c1) * (np.cos(ang1) / np.cos(ang2)) ** 2
                D2 = r1 + r2 * (self.c2 / self.c1)

                # Velocity components
                phase_term = np.exp(1j * self.k1 * r1 + 1j * self.k2 * r2)
                vx += T * px * dir_term * phase_term / np.sqrt(D1 * D2)
                vy += T * py * dir_term * phase_term / np.sqrt(D1 * D2)
                vz += T * pz * dir_term * phase_term / np.sqrt(D1 * D2)

        # External factor
        factor = (-1j * self.k1 * dx * dy) / (2 * np.pi)
        vx *= factor
        vy *= factor
        vz *= factor

        return vx, vy, vz

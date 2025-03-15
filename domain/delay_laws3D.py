# domain/delay_laws3D.py

import numpy as np
import math

class DelayLaws3D:
    """
    Class to compute time delays for a 2D array (M x N) of elements.

    Parameters:
        M (int): Number of elements in the x-direction.
        N (int): Number of elements in the y-direction.
        sx (float): Pitch in x-direction (mm).
        sy (float): Pitch in y-direction (mm).
        theta (float): Steering angle theta (degrees).
        phi (float): Steering angle phi (degrees).
        F (float): Focal distance (mm). Use np.inf for steering only.
        c (float): Wave speed (m/s).
    """
    
    def __init__(self, M: int, N: int, sx: float, sy: float,
                 theta: float, phi: float, F: float, c: float):
        if M < 1 or N < 1:
            raise ValueError("M and N must be >= 1")
        if c == 0:
            raise ValueError("Wave speed c cannot be zero (division by zero)")
        
        self.M = M
        self.N = N
        self.sx = sx
        self.sy = sy
        self.theta = theta
        self.phi = phi
        self.F = F
        self.c = c
    
    def calculate(self) -> np.ndarray:
        """
        Calculate the time delays (in microseconds) for the 2D array.

        Returns:
            np.ndarray: A 2D array (shape: M x N) of time delays.
        """
        # Element indices: m = 1,...,M and n = 1,...,N.
        m = np.arange(1, self.M + 1)
        n = np.arange(1, self.N + 1)
        Mb = (self.M - 1) / 2.0
        Nb = (self.N - 1) / 2.0
        
        # Calculate centroids of each element (in mm)
        exm = (m - 1 - Mb) * self.sx  # x-direction centroids
        eyn = (n - 1 - Nb) * self.sy  # y-direction centroids
        
        # Initialize delay matrix (in microseconds)
        dt = np.zeros((self.M, self.N), dtype=float)
        
        # Steering only mode
        if np.isinf(self.F):
            for mm in range(self.M):
                for nn in range(self.N):
                    dt[mm, nn] = 1000.0 * (
                        exm[mm] * math.sin(math.radians(self.theta)) * math.cos(math.radians(self.phi)) +
                        eyn[nn] * math.sin(math.radians(self.theta)) * math.sin(math.radians(self.phi))
                    ) / self.c
            # Shift delays so that all are positive (add absolute value of minimum)
            dt = abs(dt.min()) + dt
        else:
            # Steering + focusing mode
            r = np.zeros((self.M, self.N), dtype=float)
            for mm in range(self.M):
                for nn in range(self.N):
                    term1 = (self.F * math.sin(math.radians(self.theta)) * math.cos(math.radians(self.phi)) - exm[mm]) ** 2
                    term2 = (self.F * math.sin(math.radians(self.theta)) * math.sin(math.radians(self.phi)) - eyn[nn]) ** 2
                    term3 = (self.F * math.cos(math.radians(self.theta))) ** 2
                    r[mm, nn] = math.sqrt(term1 + term2 + term3)
            # Compute delays based on the difference from the maximum delay
            dt = 1000.0 * r / self.c
            dt = dt.max() - dt
        
        return dt

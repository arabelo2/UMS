# domain/T_fluid_solid.py

import numpy as np

class FluidSolidTransmission:
    """
    Computes P-P (tpp) and P-S (tps) transmission coefficients for a plane fluid-solid interface.
    """
    @staticmethod
    def compute_coefficients(d1, cp1, d2, cp2, cs2, theta1):
        """
        Compute transmission coefficients based on velocity ratios.

        Parameters:
            d1 (float): Density of the fluid (gm/cm^3)
            cp1 (float): Compressional wave speed in the fluid (m/s)
            d2 (float): Density of the solid (gm/cm^3)
            cp2 (float): Compressional wave speed in the solid (m/s)
            cs2 (float): Shear wave speed in the solid (m/s)
            theta1 (float): Incident angle in degrees

        Returns:
            tuple: (tpp, tps) Transmission coefficients
        """
        # Validate inputs
        if any(param <= 0 for param in [d1, cp1, d2, cp2, cs2]):
            raise ValueError("Densities and wave speeds must be positive values.")
        
        if not (0 <= theta1 <= 90):
            raise ValueError("Incident angle must be between 0 and 90 degrees.")
        
        # Convert angle to radians
        iang = np.radians(theta1)
        
        # Compute sin(theta) for refracted P- and S-waves
        sinp = (cp2 / cp1) * np.sin(iang)
        sins = (cs2 / cp1) * np.sin(iang)

        # Handle total internal reflection with explicit complex handling
        cosp = np.where(sinp >= 1, 1j * np.sqrt(sinp ** 2 - 1 + 0j), np.sqrt(1 - sinp ** 2 + 0j))
        coss = np.where(sins >= 1, 1j * np.sqrt(sins ** 2 - 1 + 0j), np.sqrt(1 - sins ** 2 + 0j))
        
        # Compute denominator with complex-safe operations
        sqrt_term = np.sqrt(1 - np.sin(iang) ** 2 + 0j)
        term1 = 4 * ((cs2 / cp2) ** 2) * (sins * coss * sinp * cosp)
        term2 = 1 - 4 * (sins ** 2) * (coss ** 2)
        denom = cosp + (d2 / d1) * (cp2 / cp1) * sqrt_term * (term1 + term2)
        
        # Avoid division by zero
        denom = np.where(np.abs(denom) < 1e-10, np.inf, denom)

        # Ensure numerators are complex and compute coefficients
        numerator_tpp = (2 * sqrt_term * (1 - 2 * (sins ** 2))).astype(np.complex128)
        numerator_tps = (-4 * cosp * sins * sqrt_term).astype(np.complex128)
        
        tpp = numerator_tpp / denom
        tps = numerator_tps / denom
        
        return tpp, tps
    
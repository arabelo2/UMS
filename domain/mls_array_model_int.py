# domain/mls_array_model_int.py

import numpy as np

class MLSArrayModelInt:
    """
    Class to compute the normalized pressure wave field for an array of 1-D elements 
    radiating waves through a fluid/fluid interface using a 2D integration approach (ls_2Dint).
    """
    def __init__(self, f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, x=None, z=None):
        """
        Initialize the MLS Array Modeling parameters.
        
        Parameters:
            f      (float): Frequency in MHz.
            d1     (float): Density of first medium (gm/cm^3).
            c1     (float): Wave speed in first medium (m/s).
            d2     (float): Density of second medium (gm/cm^3).
            c2     (float): Wave speed in second medium (m/s).
            M      (int)  : Number of elements.
            d      (float): Element length (mm).
            g      (float): Gap length (mm).
            angt   (float): Angle of array (degrees).
            ang20  (float): Steering angle in second medium (degrees).
            DF     (float): Focal depth in second medium (mm); use inf for no focusing.
            DT0    (float): Distance from array to interface (mm); must be > 0.
            wtype  (str)  : Type of amplitude weighting function.
            x      (array-like, optional): x-coordinates for field calculations.
                                           Defaults to linspace(-5,15,200) if not provided.
            z      (array-like, optional): z-coordinates for field calculations.
                                           Defaults to linspace(1,20,200) if not provided.
        """
        self.f = f
        self.d1 = d1
        self.c1 = c1
        self.d2 = d2
        self.c2 = c2
        self.M = M
        self.d = d
        self.g = g
        self.angt = angt
        self.ang20 = ang20
        self.DF = DF
        self.DT0 = DT0
        self.wtype = wtype
        
        if x is None or z is None:
            self.x = np.linspace(-5, 15, 200)
            self.z = np.linspace(1, 20, 200)
        else:
            self.x = x
            self.z = z

    def compute_field(self):
        """
        Compute the normalized pressure field using ls_2Dint.
        
        Returns:
            dict: Contains the pressure field ('p') and the grid coordinates ('x', 'z').
        """
        xx, zz = np.meshgrid(self.x, self.z)
        b = self.d / 2.0             # Element half-length.
        s = self.d + self.g          # Pitch of the array.
        mat = [self.d1, self.c1, self.d2, self.c2]  # Material properties.
        
        # Calculate element center positions.
        e = np.array([s * (((2 * mm - 1) / 2) - (self.M / 2)) for mm in range(1, self.M + 1)])
        
        # Compute time delays using the application service.
        from application.delay_laws2D_int_service import DelayLaws2DInterfaceService
        delay_service = DelayLaws2DInterfaceService()
        td = delay_service.compute_delays(self.M, s, self.angt, self.ang20, self.DT0, self.DF, self.c1, self.c2, 'n')
        delay = np.exp(1j * 2 * np.pi * self.f * td)
        
        # Retrieve discrete window amplitudes.
        from application.discrete_windows_service import run_discrete_windows_service
        Ct = run_discrete_windows_service(self.M, self.wtype)
        
        # Initialize the pressure field.
        p = np.zeros_like(xx, dtype=complex)
        
        from application.ls_2Dint_service import run_ls_2Dint_service
        for mm in range(self.M):
            p += Ct[mm] * delay[mm] * run_ls_2Dint_service(b, self.f, self.c1, e[mm], mat, self.angt, self.DT0, xx, zz, 1)
        
        return {'p': p, 'x': self.x, 'z': self.z}

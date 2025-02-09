# application/ls_2Dint_service.py

"""
Module: ls_2Dint_service.py
Layer: Application

Provides a service that wraps the LS2DInterface domain class.
"""

from domain.ls_2Dint import LS2DInterface

class LS2DInterfaceService:
    """
    Service for computing the normalized pressure using LS2DInterface.
    """
    def __init__(self, b, f, mat, e, angt, Dt0, x, z, Nopt=None):
        self._ls2d = LS2DInterface(b, f, mat, e, angt, Dt0, x, z, Nopt)

    def calculate(self):
        """
        Compute the normalized pressure.

        Returns:
            p (complex): The normalized pressure.
        """
        return self._ls2d.compute()

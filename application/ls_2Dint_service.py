# application/ls_2Dint_service.py

from domain.ls_2Dint import ls_2Dint

def run_ls_2Dint_service(b, f, c, e, mat, angt, Dt0, x, z, Nopt=None):
    """
    Service function for ls_2Dint.
    
    Parameters:
      (Same as ls_2Dint)
      
    Returns:
      p: The computed normalized pressure.
    """
    return ls_2Dint(b, f, c, e, mat, angt, Dt0, x, z, Nopt)

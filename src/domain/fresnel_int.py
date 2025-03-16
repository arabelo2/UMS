# domain/fresnel_int.py

import numpy as np
from domain.cs_int import cs_int

def fresnel_int(x):
    """
    Compute the Fresnel integral defined as the integral from 0 to x of exp(i*pi*t^2/2).
    
    This function implements an approximate evaluation of the Fresnel integral using the 
    expressions given by Abramowitz and Stegun, Handbook of Mathematical Functions, Dover Publications, 1965, pp. 301-302.
    
    The computation proceeds by:
      1. Separating the input x into negative and non-negative parts.
         For negative values, the sign is reversed (xn = -x for x < 0).
      2. Computing the cosine and sine integrals for the negative values using cs_int,
         then reversing their signs (because the integrals are odd functions).
      3. Computing the cosine and sine integrals for the non-negative values directly.
      4. Combining the results from the negative and non-negative parts:
            ct = [cn, cp] and st = [sn, sp],
         and returning the complex Fresnel integral as:
            y = ct + 1j * st.
    
    Parameters:
        x : float or numpy array
            The upper limit(s) of integration (in arbitrary units). x may be a scalar or a numpy array.
    
    Returns:
        y : complex or numpy array of complex numbers
            The computed Fresnel integral. If the input is a scalar, y is returned as a scalar.
    """
    # Ensure x is a numpy array.
    x = np.asarray(x, dtype=float)
    
    # Create boolean masks for negative and non-negative values.
    neg_mask = x < 0
    pos_mask = ~neg_mask  # equivalent to x >= 0
    
    # For negative values, compute xn = -x.
    xn = -x[neg_mask]
    xp = x[pos_mask]
    
    # Compute cs_int for negative values and reverse the sign.
    if xn.size > 0:
        cn, sn = cs_int(xn)
        cn = -cn
        sn = -sn
    else:
        cn = np.array([])
        sn = np.array([])
    
    # Compute cs_int for non-negative values.
    if xp.size > 0:
        cp, sp = cs_int(xp)
    else:
        cp = np.array([])
        sp = np.array([])
    
    # Combine the results.
    # The MATLAB code concatenates the results: first for negative values, then positive.
    ct = np.concatenate((cn, cp))
    st = np.concatenate((sn, sp))
    
    # Form the complex Fresnel integral.
    y = ct + 1j * st
    
    # If the input was scalar, return a scalar.
    if y.size == 1:
        return y.item()
    return y

# domain/interface2.py

from dataclasses import dataclass
import cmath

@dataclass(frozen=True)
class Interface2Parameters:
    """
    Holds the parameters for the interface function:
    - cr: c1/c2 ratio (wave speed ratio between medium one and medium two)
    - df: depth of the point in medium two (DF)
    - dp: height of the point in medium one (DT)
    - dpf: separation distance between the two points (DX)
    """
    cr: float
    df: float
    dp: float
    dpf: float

class Interface2:
    """
    Domain class to evaluate the interface function defined by:

      y = x/sqrt(x^2 + dp^2) - cr*(dpf - x)/sqrt((dpf - x)^2 + df^2)
      
    where x is the location along the interface and the other parameters are provided.
    """
    def __init__(self, parameters: Interface2Parameters):
        self._parameters = parameters

    def evaluate(self, x: float) -> float:
        """
        Evaluate the function at a given x.
        
        Parameters:
            x (float): The location along the interface.
            
        Returns:
            float: The computed value.
        """
        dp = self._parameters.dp
        df = self._parameters.df
        cr = self._parameters.cr
        dpf = self._parameters.dpf

        term1 = x / cmath.sqrt(x ** 2 + dp ** 2)
        term2 = (dpf - x) / cmath.sqrt((dpf - x) ** 2 + df ** 2)
        return term1 - cr * term2

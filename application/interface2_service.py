# application/interface2_service.py

from domain.interface2 import Interface2, Interface2Parameters

class Interface2Service:
    """
    Application service that acts as a mediator between the interface (user input)
    and the domain logic. It receives an instance of the parameters, creates the domain
    object, and provides a method to compute the function.
    """
    def __init__(self, parameters: Interface2Parameters):
        self._interface2 = Interface2(parameters)

    def compute(self, x: float) -> float:
        """
        Computes the function value at the given x.
        
        Parameters:
            x (float): The location along the interface.
            
        Returns:
            float: The computed function value.
        """
        return self._interface2.evaluate(x)

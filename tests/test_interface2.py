# tests/test_interface2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import unittest
from domain.interface2 import Interface2Parameters, Interface2
from application.interface2_service import Interface2Service

class TestInterface2(unittest.TestCase):
    def setUp(self):
        # Use sample parameters for testing.
        self.parameters = Interface2Parameters(cr=1.5, df=10.0, dp=5.0, dpf=20.0)
        self.interface2 = Interface2(self.parameters)
        self.service = Interface2Service(self.parameters)

    def test_evaluate_zero(self):
        """
        Test the function evaluation at x = 0.
        """
        x = 0.0
        expected = (
            0.0 / math.sqrt(0.0 ** 2 + self.parameters.dp ** 2) -
            self.parameters.cr * (self.parameters.dpf) /
            math.sqrt(self.parameters.dpf ** 2 + self.parameters.df ** 2)
        )
        result = self.interface2.evaluate(x)
        self.assertAlmostEqual(result, expected, places=7)

    def test_evaluate_arbitrary(self):
        """
        Test the function evaluation at an arbitrary x.
        """
        x = 5.0
        expected = (
            x / math.sqrt(x ** 2 + self.parameters.dp ** 2) -
            self.parameters.cr * (self.parameters.dpf - x) /
            math.sqrt((self.parameters.dpf - x) ** 2 + self.parameters.df ** 2)
        )
        result = self.interface2.evaluate(x)
        self.assertAlmostEqual(result, expected, places=7)

    def test_service_compute(self):
        """
        Test that the service returns the same result as the domain logic.
        """
        x = 5.0
        expected = self.interface2.evaluate(x)
        result = self.service.compute(x)
        self.assertAlmostEqual(result, expected, places=7)

if __name__ == "__main__":
    unittest.main()

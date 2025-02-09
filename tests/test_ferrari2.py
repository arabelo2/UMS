# tests/test_ferrari2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from application.ferrari2_service import Ferrari2Service

class TestFerrari2(unittest.TestCase):
    def test_identical_media(self):
        # When cr is nearly 1, the explicit formula is used.
        cr = 1.0
        DF = 10.0
        DT = 5.0
        DX = 20.0
        service = Ferrari2Service(cr, DF, DT, DX)
        xi = service.solve()
        expected = DX * DT / (DF + DT)
        self.assertAlmostEqual(xi, expected, places=6)

    def test_ferrari_method(self):
        # Test with parameters that force use of Ferrari's method.
        cr = 1.5
        DF = 10.0
        DT = 5.0
        DX = 20.0
        service = Ferrari2Service(cr, DF, DT, DX)
        xi = service.solve()
        # Since an analytic expected value is not available,
        # we check that the solution lies within [0, DX] (for DX positive).
        self.assertTrue(0 <= xi <= DX)

    def test_dx_zero(self):
        # Test for different media (cr ≠ 1) when DX = 0.
        # When the separation along the interface is zero, we expect xi to be zero.
        cr = 1.5
        DF = 10.0
        DT = 5.0
        DX = 0.0
        service = Ferrari2Service(cr, DF, DT, DX)
        xi = service.solve()
        expected = 0.0
        self.assertAlmostEqual(xi, expected, places=6)

if __name__ == "__main__":
    unittest.main()

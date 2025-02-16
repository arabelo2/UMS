# tests/test_elements.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from domain.elements import ElementsCalculator

class TestElementsCalculator(unittest.TestCase):

    def test_valid_calculation(self):
        calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0.1, num_elements=4)
        A, d, g, xc = calc.calculate()
        self.assertAlmostEqual(d, 0.148, places=3)
        self.assertAlmostEqual(g, 0.0148, places=4)
        self.assertAlmostEqual(A, 0.6364, places=4)
        self.assertEqual(len(xc), 4)

    def test_zero_frequency(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=0, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0.1, num_elements=4)
            calc.calculate()

    def test_negative_elements(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0.1, num_elements=-1)
            calc.calculate()

    def test_large_array(self):
        calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0.1, num_elements=1000)
        A, d, g, xc = calc.calculate()
        self.assertEqual(len(xc), 1000)

    def test_edge_case_gap_ratio(self):
        calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0, num_elements=5)
        A, d, g, xc = calc.calculate()
        self.assertEqual(g, 0)
        self.assertEqual(len(xc), 5)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=-5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=0.1, num_elements=4)
            calc.calculate()

    def test_zero_wave_speed(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=0, diameter_ratio=0.5, gap_ratio=0.1, num_elements=4)
            calc.calculate()

    def test_invalid_gap_ratio(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=0.5, gap_ratio=-0.1, num_elements=4)
            calc.calculate()

    def test_invalid_diameter_ratio(self):
        with self.assertRaises(ValueError):
            calc = ElementsCalculator(frequency_mhz=5, wave_speed_m_s=1480, diameter_ratio=-0.5, gap_ratio=0.1, num_elements=4)
            calc.calculate()

if __name__ == '__main__':
    unittest.main()

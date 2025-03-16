# domain/elements.py

import numpy as np

class ElementsCalculator:
    def __init__(self, frequency_mhz, wave_speed_m_s, diameter_ratio, gap_ratio, num_elements):
        self.frequency_mhz = frequency_mhz
        self.wave_speed_m_s = wave_speed_m_s
        self.diameter_ratio = diameter_ratio
        self.gap_ratio = gap_ratio
        self.num_elements = num_elements

    def calculate(self):
        # Validate parameters
        if self.frequency_mhz <= 0:
            raise ValueError("Frequency must be greater than zero.")
        if self.wave_speed_m_s <= 0:
            raise ValueError("Wave speed must be greater than zero.")
        if self.diameter_ratio <= 0:
            raise ValueError("Diameter ratio must be positive.")
        if self.gap_ratio < 0:
            raise ValueError("Gap ratio cannot be negative.")
        if self.num_elements <= 0:
            raise ValueError("Number of elements must be positive.")

        # Calculate element diameter
        d = self.diameter_ratio * self.wave_speed_m_s / (1000 * self.frequency_mhz)

        # Calculate gap size
        g = self.gap_ratio * d

        # Calculate total aperture size
        A = self.num_elements * d + (self.num_elements - 1) * g

        # Calculate centroids of the array elements
        Mb = (self.num_elements - 1) / 2
        xc = np.array([(g + d) * ((2 * nn - 1) / 2 - self.num_elements / 2) for nn in range(1, self.num_elements + 1)])

        return A, d, g, xc

import numpy as np

class InterfaceFunction:
    def __init__(self, wave_speed_ratio, depth, height, separation_distance):
        self.wave_speed_ratio = wave_speed_ratio
        self.depth = depth
        self.height = height
        self.separation_distance = separation_distance

    def calculate_y(self, x):
        term1 = x / np.sqrt(x**2 + self.height**2)
        term2 = self.wave_speed_ratio * (self.separation_distance - x) / np.sqrt((self.separation_distance - x)**2 + self.depth**2)
        return term1 - term2

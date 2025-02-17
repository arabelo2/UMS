# domain/mls_array_modeling.py

class MLSArrayModeling:
    def __init__(self, f, c, M, dl, gd, Phi, F, wtype):
        self.frequency = f
        self.wave_speed = c
        self.num_elements = M
        self.diameter_ratio = dl
        self.gap_ratio = gd
        self.steering_angle = Phi
        self.focal_length = F
        self.window_type = wtype
        self.window_amplitudes = None
        self.time_delays = None

    def calculate_elements(self):
        from domain.elements import ElementsCalculator
        calculator = ElementsCalculator(
            frequency_mhz=self.frequency,
            wave_speed_m_s=self.wave_speed,
            diameter_ratio=self.diameter_ratio,
            gap_ratio=self.gap_ratio,
            num_elements=self.num_elements
        )
        return calculator.calculate()

    def calculate_time_delays(self, s):
        from application.delay_laws2D_service import run_delay_laws2D_service
        self.time_delays = run_delay_laws2D_service(
            self.num_elements, s, self.steering_angle, self.focal_length, self.wave_speed
        )
        return self.time_delays

    def calculate_window_amplitudes(self):
        from application.discrete_windows_service import run_discrete_windows_service
        self.window_amplitudes = run_discrete_windows_service(self.num_elements, self.window_type)
        return self.window_amplitudes

    def calculate_pressure_field(self, b, centroid, xx, zz):
        """
        Return the spatial pressure contribution from a single element located at 'centroid'.
        (Note: The window amplitude and delay factors are applied in the service layer.)
        """
        from application.ls_2Dv_service import run_ls_2Dv_service
        pressure = run_ls_2Dv_service(b, self.frequency, self.wave_speed, centroid, xx, zz)
        return pressure.squeeze()  # Remove any extra dimensions, e.g., from shape (1,500,500) to (500,500)

class InterfaceService:
    def __init__(self, interface_function):
        self.interface_function = interface_function

    def evaluate(self, x_values):
        return self.interface_function.calculate_y(x_values)

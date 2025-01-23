# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.gauss_c10_service import GaussC10Service


def main():
    # Initialize the service
    service = GaussC10Service()

    # Retrieve the coefficients
    a, b = service.get_coefficients()

    # Display the coefficients
    print("Coefficients a:")
    for idx, coeff in enumerate(a, start=1):
        print(f"  a[{idx}] = {coeff.real:.5f} + {coeff.imag:.5f}j")

    print("\nCoefficients b:")
    for idx, coeff in enumerate(b, start=1):
        print(f"  b[{idx}] = {coeff.real:.5f} + {coeff.imag:.5f}j")


if __name__ == "__main__":
    main()

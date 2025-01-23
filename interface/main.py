# interface/main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.elements_service import ElementsService


def main():
    # Input parameters
    f = 5.0  # Frequency (MHz)
    c = 1480.0  # Wave speed (m/s)
    dl = 0.5  # Element length divided by wavelength (d/λ)
    gd = 0.1  # Gap size divided by element length (g/d)
    N = 32  # Number of elements

    # Initialize the service
    service = ElementsService()

    # Compute array properties
    A, d, g, xc = service.calculate(f, c, dl, gd, N)

    # Display the results
    print(f"Total Aperture Size (A): {A:.4f} mm")
    print(f"Element Size (d): {d:.4f} mm")
    print(f"Gap Size (g): {g:.4f} mm")
    print("Centroids (xc):")
    for idx, centroid in enumerate(xc, start=1):
        print(f"  Element {idx}: {centroid:.4f} mm")


if __name__ == "__main__":
    main()

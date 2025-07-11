#!/usr/bin/env python3
"""
Module: delay_laws3Dint_interface.py
Layer: Interface

Provides a CLI to compute time delays (in microseconds) for a 2D array
steered through a planar interface using the integrated delay_laws3Dint function.
The delays are saved to an output file and, if requested, a 3D stem plot (emulating MATLAB’s stem3)
is displayed.

Default values:
  Mx         = 8         (number of elements in x-direction)
  My         = 16        (number of elements in y-direction)
  sx         = 0.5       mm (pitch in x-direction)
  sy         = 0.5       mm (pitch in y-direction)
  theta      = 0         deg (array angle with interface)
  phi        = 0         deg (steering angle for the second medium)
  theta20    = 30        deg (refracted steering angle in second medium)
  DT0        = 25.4      mm (height of array center above interface)
  DF         = 10        mm (focal distance in second medium; use inf for steering-only)
  c1         = 1480      m/s (wave speed in the first medium)
  c2         = 5900      m/s (wave speed in the second medium)
  z_scale    = 1.0             (scaling factor for the delay array)
  outfile    = "delay_laws3Dint_output.txt" (output filename)
  plot       = "Y"       (display a 3D stem plot? "Y" or "N")
  elev       = 25        (camera elevation for 3D plot)
  azim       = 20        (camera azimuth for 3D plot)

Example usage:
  python delay_laws3Dint_interface.py --Mx 8 --My 16 --sx 0.5 --sy 0.5 --theta 0 --phi 0 \
       --theta20 30 --DT0 25.4 --DF inf --c1 1480 --c2 5900 --z_scale 1.5 --outfile delay_laws3Dint_output.txt \
       --plot Y --elev 25 --azim 20
"""

import sys
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Adjust sys.path so we can import modules from the application and interface layers.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interface.cli_utils import safe_float
from application.delay_laws3Dint_service import run_delay_laws3Dint_service


class DelayLaws3DIntCLI:
    """
    Command-line interface for computing the integrated delay laws for a 2D array
    (steered/focused through a planar interface in 3D).
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Compute time delays (µs) for a 2D array using delay_laws3Dint.",
            epilog=(
                "Example usage:\n"
                "  python delay_laws3Dint_interface.py --Mx 8 --My 16 --sx 0.5 --sy 0.5 --theta 0 --phi 0 \\\n"
                "       --theta20 30 --DT0 25.4 --DF 10 --c1 1480 --c2 5900 \\\n"
                "       --z_scale 1.0 --outfile delay_laws3Dint_output.txt --plot Y --elev 25 --azim 20\n\n"
                "Note: Use DF=inf for a steering-only case."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self._setup_arguments()

    def _setup_arguments(self):
        # Array geometry and element parameters
        self.parser.add_argument("--Mx", type=int, default=8,
                                 help="Number of elements in x-direction. Default: 8")
        self.parser.add_argument("--My", type=int, default=16,
                                 help="Number of elements in y-direction. Default: 16")
        self.parser.add_argument("--sx", type=safe_float, default=0.5,
                                 help="Pitch in x-direction (mm). Default: 0.5")
        self.parser.add_argument("--sy", type=safe_float, default=0.5,
                                 help="Pitch in y-direction (mm). Default: 0.5")
        self.parser.add_argument("--theta", type=safe_float, default=0.0,
                                 help="Array angle with interface (deg). Default: 0")
        self.parser.add_argument("--phi", type=safe_float, default=0.0,
                                 help="Steering angle for the second medium (deg). Default: 0")
        # Steering / focusing parameters
        self.parser.add_argument("--theta20", type=safe_float, default=30.0,
                                 help="Refracted steering angle in second medium (deg). Default: 30")
        self.parser.add_argument("--DT0", type=safe_float, default=25.4,
                                 help="Height of array center above interface (mm). Default: 25.4")
        self.parser.add_argument("--DF", type=safe_float, default=10.0,
                                 help="Focal distance in the second medium (mm); use inf for steering-only. Default: 10")
        # Acoustic properties
        self.parser.add_argument("--c1", type=safe_float, default=1480.0,
                                 help="Wave speed in the first medium (m/s). Default: 1480")
        self.parser.add_argument("--c2", type=safe_float, default=5900.0,
                                 help="Wave speed in the second medium (m/s). Default: 5900")
        # Scaling parameter for delays
        self.parser.add_argument("--z_scale", type=safe_float, default=1.0,
                                 help="Scaling factor for the computed delay array. Default: 1.0")
        # Output file name
        self.parser.add_argument("--outfile", type=str, default="delay_laws3Dint_output.txt",
                                 help="Filename to save the computed delays. Default: delay_laws3Dint_output.txt")
        # Plot options
        self.parser.add_argument("--plot", type=lambda s: s.upper(), choices=["Y", "N"], default="Y",
                                 help="Display a 3D stem plot? (Y/N). Default: Y")
        self.parser.add_argument("--elev", type=safe_float, default=25.0,
                                 help="Camera elevation for 3D plot. Default: 25")
        self.parser.add_argument("--azim", type=safe_float, default=20.0,
                                 help="Camera azimuth for 3D plot. Default: 20")

    def parse_arguments(self):
        return self.parser.parse_args()

    def save_delays_to_file(self, delays: np.ndarray, outfile: str):
        """
        Save the computed delay array to the specified file.
        """
        Mx, My = delays.shape
        with open(outfile, "w") as f:
            for i in range(Mx):
                for j in range(My):
                    f.write(f"Element ({i+1},{j+1}): {delays[i, j]:.6f} µs\n")
        print(f"Time delays saved to {outfile}")

    def plot_delays_3d_stem(self, delays: np.ndarray, elev: float, azim: float):
        """
        Create a 3D stem plot of the delay array.
        """
        Mx, My = delays.shape
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        # Use element indices for plotting. (Rows: 1 to Mx, Columns: 1 to My)
        X, Y = np.meshgrid(np.arange(1, My + 1), np.arange(1, Mx + 1))
        for i in range(Mx):
            for j in range(My):
                ax.plot([Y[i, j], Y[i, j]], [X[i, j], X[i, j]], [0, delays[i, j]], 'b-o')
        ax.set_xlabel("Element Index (y-direction)", fontsize=20)
        ax.set_ylabel("Element Index (x-direction)", fontsize=20)
        ax.set_zlabel("Time Delay (µs)", fontsize=20)
        mode = "Steering Only" if math.isinf(self.args.DF) else "Steering + Focusing"
        title_str = (
            f"Delay Laws 3D - {mode}\n"
            f"Mx={self.args.Mx}, My={self.args.My}, sx={self.args.sx} mm, sy={self.args.sy} mm,\n"
            f"theta={self.args.theta}°, phi={self.args.phi}°, theta20={self.args.theta20}°, "
            f"DT0={self.args.DT0} mm, DF={self.args.DF} mm,\n"
            f"c1={self.args.c1} m/s, c2={self.args.c2} m/s, z_scale={self.args.z_scale}"
        )
        ax.set_title(title_str, fontsize=20)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def run(self):
        # Parse command-line arguments.
        self.args = self.parse_arguments()

        # Compute the delay laws by calling the service layer.
        try:
            delays = run_delay_laws3Dint_service(
                self.args.Mx, self.args.My,
                self.args.sx, self.args.sy,
                self.args.theta, self.args.phi, self.args.theta20,
                self.args.DT0, self.args.DF,
                self.args.c1, self.args.c2,
                self.args.plot,      # plt_option as string (e.g., 'Y' or 'N')
                self.args.elev, self.args.azim,
                self.args.z_scale    # scaling factor
            )
        except Exception as e:
            print(f"Error computing delay laws: {e}", file=sys.stderr)
            sys.exit(1)

        # Output the computed delays to stdout.
        print("Computed time delays (µs):")
        print(delays)

        # Save delays to file.
        self.save_delays_to_file(delays, self.args.outfile)

        # Create a 3D stem plot if requested.
        if self.args.plot == "Y":
            self.plot_delays_3d_stem(delays, self.args.elev, self.args.azim)


def main():
    cli = DelayLaws3DIntCLI()
    cli.run()


if __name__ == "__main__":
    main()

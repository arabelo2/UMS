#!/usr/bin/env python
# interface/mls_array_model_int_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from interface.cli_utils import safe_float, parse_array
from application.ml s_array_model_int_service import run_mls_array_model_int_service  # Ensure module name is correct

def main():
    parser = argparse.ArgumentParser(
        description="Simulate the MLS Array Modeling process at a fluid/fluid interface using ls_2Dint.",
        epilog=(
            "Example usage:\n"
            "  python interface/mls_array_model_int_interface.py --f 5 --d1 1.0 --c1 1480 --d2 7.9 --c2 5900 "
            "--M 32 --d 0.25 --g 0.05 --angt 0 --ang20 30 --DF 8 --DT0 25.4 --wtype rect --plot Y\n"
            "Defaults simulate the MLS Array Modeling at the interface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--f", type=safe_float, default=5, help="Frequency in MHz. Default: 5")
    parser.add_argument("--d1", type=safe_float, default=1.0, help="Density of first medium (gm/cm^3). Default: 1.0")
    parser.add_argument("--c1", type=safe_float, default=1480, help="Wave speed in first medium (m/s). Default: 1480")
    parser.add_argument("--d2", type=safe_float, default=7.9, help="Density of second medium (gm/cm^3). Default: 7.9")
    parser.add_argument("--c2", type=safe_float, default=5900, help="Wave speed in second medium (m/s). Default: 5900")
    parser.add_argument("--M", type=int, default=32, help="Number of elements. Default: 32")
    parser.add_argument("--d", type=safe_float, default=0.25, help="Element length (mm). Default: 0.25")
    parser.add_argument("--g", type=safe_float, default=0.05, help="Gap length (mm). Default: 0.05")
    parser.add_argument("--angt", type=safe_float, default=0, help="Angle of array (degrees). Default: 0")
    parser.add_argument("--ang20", type=safe_float, default=30, help="Steering angle in second medium (degrees). Default: 30")
    parser.add_argument("--DF", type=safe_float, default=8, help="Focal depth in second medium (mm). Default: 8")
    parser.add_argument("--DT0", type=safe_float, default=25.4, help="Distance from array to interface (mm). Default: 25.4")
    parser.add_argument("--wtype", type=str, default="rect", choices=["cos", "Han", "Ham", "Blk", "tri", "rect"],
                        help="Amplitude weighting function type. Default: rect")
    parser.add_argument("--plot", type=str, choices=["Y", "N"], default="Y", help="Plot the result? Y/N. Default: Y")
    
    args = parser.parse_args()
    
    result = run_mls_array_model_int_service(
         args.f, args.d1, args.c1, args.d2, args.c2,
         args.M, args.d, args.g, args.angt, args.ang20,
         args.DF, args.DT0, args.wtype
    )
    
    p = result['p']
    x = result['x']
    z = result['z']
    
    outfile = "mls_array_model_int_output.txt"
    with open(outfile, "w") as f:
         for i in range(p.shape[0]):
             for j in range(p.shape[1]):
                 f.write(f"{p[i, j].real:.6f}+{p[i, j].imag:.6f}j\n")
    print(f"Results saved to {outfile}")
    
    if args.plot.upper() == "Y":
         plt.figure(figsize=(10, 6))
         plt.imshow(np.abs(p), cmap="jet", extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto')
         plt.xlabel("x (mm)")
         plt.ylabel("z (mm)")
         plt.title("MLS Array Modeling Pressure Field at Fluid/Fluid Interface")
         plt.colorbar(label="Pressure Magnitude")
         plt.tight_layout()
         plt.show()

if __name__ == "__main__":
    main()

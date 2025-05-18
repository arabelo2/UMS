#!/usr/bin/env python3
# interface/beam_model_interface.py
#
# Unified front‑end for LS‑2Dv (2‑D line source)  *or*
# Rectangular‑piston PS‑3Dv (3‑D) pressure models.
#
# Usage examples
# --------------
# 2‑D beam line source:
#   python interface/beam_model_interface.py --model ls2Dv --b 0.5 --ex 0 \
#       --f 5 --c 1480 --plot Y
#
# 3‑D rectangular piston:
#   python interface/beam_model_interface.py --model ps3Dv --lx 0.6 --ly 0.6 \
#       --ex 0 --ey 0 --f 5 --c 1480 --plot Y
#
# -------------------------------------------------------------------------

import os, sys, argparse, numpy as np, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interface.cli_utils import safe_float, parse_array
from application.ls_2Dv_service import run_ls_2Dv_service
from application.ps_3Dv_service import run_ps_3Dv_service

# -------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified LS‑2Dv / PS‑3Dv beam‑model interface."
    )
    parser.add_argument("--model", required=True,
                        choices=["ls2Dv", "ps3Dv"],
                        help="Choose the underlying model.")
    parser.add_argument("--f",  type=safe_float, default=5,
                        help="Frequency [MHz].")
    parser.add_argument("--c",  type=safe_float, default=1480,
                        help="Wave speed [m/s].")
    # --------------- LS‑2Dv specific ----------------
    parser.add_argument("--b",  type=safe_float,
                        help="[LS‑2Dv] Half‑length b [mm].")
    parser.add_argument("--ex", type=safe_float, default=0,
                        help="Lateral offset x [mm] (ls2Dv) or x‑offset of piston centre (ps3Dv).")
    # --------------- PS‑3Dv specific ----------------
    parser.add_argument("--lx", type=safe_float,
                        help="[PS‑3Dv] Element length along x [mm].")
    parser.add_argument("--ly", type=safe_float,
                        help="[PS‑3Dv] Element length along y [mm].")
    parser.add_argument("--ey", type=safe_float, default=0,
                        help="[PS‑3Dv] Lateral offset y [mm].")
    # --------------- Discretisation overrides -------
    parser.add_argument("--N",  type=int, help="[LS‑2Dv] #segments.")
    parser.add_argument("--P",  type=int, help="[PS‑3Dv] #segments x.")
    parser.add_argument("--Q",  type=int, help="[PS‑3Dv] #segments y.")
    # --------------- Mesh grids ---------------------
    parser.add_argument("--x",  type=parse_array,
                        default="-20,20,401", help="x range (start,stop,pts).")
    parser.add_argument("--z",  type=parse_array,
                        default="  5,80,401", help="z range (start,stop,pts).")
    parser.add_argument("--y",  type=parse_array,
                        default="0", help="[PS‑3Dv] y slice(s) (comma list).")
    # --------------- Plotting -----------------------
    parser.add_argument("--plot", choices=["Y", "N"], default="Y",
                        help="Display result.")
    args = parser.parse_args()

    # ------------------------------------------------
    # Build pressure field
    # ------------------------------------------------
    if args.model == "ls2Dv":
        if args.b is None:
            parser.error("--b is required for ls2Dv.")
        xx, zz = np.meshgrid(args.x, args.z)
        p = run_ls_2Dv_service(
                b=args.b,
                f=args.f, c=args.c,
                e=args.ex,
                x=xx, z=zz,
                N=args.N
            )
        outfile = "ls2Dv_pressure.txt"

    else:  # ps3Dv
        if args.lx is None or args.ly is None:
            parser.error("--lx and --ly are required for ps3Dv.")
        xx, zz = np.meshgrid(args.x, args.z)           # x‑z slice(s)
        yy = np.atleast_1d(args.y)
        if yy.size != 1:
            parser.error("For quick plotting, supply a single y value.")
        yy = yy[0] * np.ones_like(xx)
        p = run_ps_3Dv_service(
                lx=args.lx, ly=args.ly,
                f=args.f, c=args.c,
                ex=args.ex, ey=args.ey,
                x=xx, y=yy, z=zz,
                P=args.P, Q=args.Q
            )
        outfile = "ps3Dv_pressure.txt"

    # ------------------------------------------------
    # Save to disk
    # ------------------------------------------------
    np.savetxt(outfile, np.column_stack([xx.ravel(), zz.ravel(),
                                         p.real.ravel(), p.imag.ravel()]),
               header="x(mm) z(mm) Re{p} Im{p}")
    print(f"Pressure field saved to {outfile}")

    # ------------------------------------------------
    # Optional plot
    # ------------------------------------------------
    if args.plot.upper() == "Y":
        plt.figure(figsize=(7, 5))
        pc = plt.pcolormesh(args.x, args.z,
                            20*np.log10(np.abs(p) / np.max(np.abs(p))),
                            cmap="jet", shading="auto", vmin=-40, vmax=0)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x [mm]")
        plt.ylabel("z [mm]")
        plt.title(f"{args.model}  |  f = {args.f} MHz")
        plt.colorbar(pc, label="Amplitude [dB]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# src/interface/master_pipeline.py
"""
master_pipeline.py – Complete Digital Twin → FMC → TFM workflow
"""

import sys
import os
import numpy as np
import argparse
import json
from scipy.signal import hilbert

# --------------------------------------------------------------------------
#  Local package path
# --------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ─── Application-layer imports ─────────────────────────────────────────────
from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from application.mps_array_model_int_service import run_mps_array_model_int_service
from application.pts_3Dintf_service import run_pts_3Dintf_service   # interface-aware FMC
from application.discrete_windows_service import run_discrete_windows_service  # <-- NEW
from interface.cli_utils import safe_float
# --------------------------------------------------------------------------

# ───────────────────────── Utility helpers ────────────────────────────────
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_run_params(params, out_root: str) -> None:
    """Persist runtime arguments for reproducibility."""
    with open(os.path.join(out_root, "run_params.json"), "w") as f:
        json.dump(vars(params), f, indent=4)
    print(f"[AP] Saved run parameters → {out_root}/run_params.json")

def parse_scan_vector(input_val, default_start, default_stop, default_num):
    """Parse a MATLAB-style vector spec (e.g. '0:0.5:10' or '0,10,100')."""
    if input_val is None:
        return np.linspace(default_start, default_stop, default_num)
    if not isinstance(input_val, str):
        return np.array(input_val, dtype=float)

    s = input_val.strip()
    if ':' in s:                               # a:b:c form → arange
        a, b, c = [float(x) for x in s.split(':')]
        return np.arange(a, c + b, b)
    nums = [float(p) for p in s.split(',') if p]
    return (np.linspace(nums[0], nums[1], int(nums[2]))
            if len(nums) == 3 else np.array(nums))

def save_data(folder: str, name: str, arrays: dict, fmt: str) -> None:
    """Save NumPy arrays as .npz or CSV files (handles 3-D cubes)."""
    if fmt == 'npz':
        np.savez(os.path.join(folder, f"{name}.npz"), **arrays)
    else:
        for key, arr in arrays.items():
            base = os.path.join(folder, f"{name}_{key}")
            if arr.ndim == 3:
                for k in range(arr.shape[2]):
                    np.savetxt(f"{base}_z{k}.csv", arr[:, :, k], delimiter=',')
            else:
                np.savetxt(f"{base}.csv", np.atleast_2d(arr), delimiter=',')
    print(f"[AP] Saved data → {folder}/{name}.*")

def calculate_envelope_fwhm(signal: np.ndarray, z_vals: np.ndarray):
    """Return analytic-signal envelope and axial FWHM (mm)."""
    env = np.abs(hilbert(np.abs(signal)))
    peak = np.argmax(env)
    half = env[peak] / 2
    mask = env > half
    edges = np.where(np.diff(mask))[0]
    if len(edges) >= 2:
        def x_at(edge):
            x0, x1 = z_vals[edge], z_vals[edge+1]
            y0, y1 = env[edge]-half, env[edge+1]-half
            return x0 - y0*(x1-x0)/(y1-y0)
        fwhm = x_at(edges[-1]) - x_at(edges[0])
        return env, fwhm
    return env, 0.0
# --------------------------------------------------------------------------

# ───────────────────── Digital-twin field simulation ──────────────────────
def run_digital_twin_field(params, out_root: str, fmt: str):
    out_dir = os.path.join(out_root, "digital_twin")
    ensure_dir(out_dir)

    xs = parse_scan_vector(params.xs, -5, 20, 100)
    zs = parse_scan_vector(params.zs,  1, 20, 100)
    ys = parse_scan_vector(params.y_vec, 0,  0,   1)

    result = run_mps_array_model_int_service(
        params.lx, params.ly, params.gx, params.gy,
        params.f,  params.d1, params.c1,
        params.d2, params.c2, params.cs2,
        params.wave_type,
        params.L1, params.L2,
        params.angt, params.Dt0,
        params.theta20, params.phi, params.DF,
        params.ampx_type, params.ampy_type,
        xs, zs, ys
    )

    save_data(out_dir, "field", {
        "p_field": np.abs(result["p"]),
        "x_vals": xs, "y_vals": ys, "z_vals": zs
    }, fmt)

# ───────────────────── FMC + TFM through interface ────────────────────────
def run_fmc_tfm(params, z_mm: str, out_root: str, fmt: str):
    out_dir = os.path.join(out_root, "fmc_tfm")
    ensure_dir(out_dir)

    # Delay laws (µs) for P- or S-wave focusing/steering
    td = run_delay_laws3Dint_service(
        params.L1, params.L2,
        params.lx + params.gx, params.ly + params.gy,
        params.angt, params.phi, params.theta20,
        params.Dt0, params.DF,
        params.c1, params.c2,
        'n'
    )

    z_vals = parse_scan_vector(z_mm, 1, 20, 100)          # axial scan (mm)
    M, N = params.L1, params.L2
    x_elem = (np.arange(M) - (M-1)/2) * (params.lx + params.gx)
    y_elem = (np.arange(N) - (N-1)/2) * (params.ly + params.gy)

    # --- NEW: pre-compute apodisation vectors for imaging path -------------
    ampx = run_discrete_windows_service(M, params.ampx_type)
    ampy = run_discrete_windows_service(N, params.ampy_type)
    # ----------------------------------------------------------------------

    FMC = np.zeros((M, N, len(z_vals)), dtype=complex)

    # Compute each Tx/Rx A-scan with full fluid–solid interface physics
    for tx in range(M):
        for rx in range(N):
            scan = run_pts_3Dintf_service(
                ex=x_elem[tx], ey=y_elem[rx],   # element offsets (mm)
                xn=tx, yn=rx,                  # indices (for apodisation)
                angt=params.angt, Dt0=params.Dt0,
                c1=params.c1, c2=params.c2,
                x=0.0, y=0.0, z=z_vals         # field point on axis
            )

            # --- NEW: apply apodisation weight ONCE to the A-scan ----------
            scan *= ampx[tx] * ampy[rx]
            # ----------------------------------------------------------------

            FMC[tx, rx, :] = scan

    # ─── Delay-and-sum TFM back-projection ────────────────────────────────
    tfm_raw = np.zeros(len(z_vals), dtype=complex)
    s_idx = np.arange(len(z_vals))

    for tx in range(M):
        for rx in range(N):
            delays = (td[tx, rx, :] if td.ndim == 3 else
                      np.full(len(z_vals), td[tx, rx] if td.ndim == 2 else td[tx]))
            # real & imag interpolated separately
            sr = np.interp(s_idx + delays, s_idx, FMC[tx, rx, :].real, left=0, right=0)
            si = np.interp(s_idx + delays, s_idx, FMC[tx, rx, :].imag, left=0, right=0)
            tfm_raw += sr + 1j*si

    envelope, fwhm = calculate_envelope_fwhm(np.abs(tfm_raw), z_vals)

    save_data(out_dir, "results", {
        "delays": td,
        "fmc_real": FMC.real, "fmc_imag": FMC.imag,
        "tfm_raw_real": tfm_raw.real, "tfm_raw_imag": tfm_raw.imag,
        "envelope": envelope, "fwhm": np.array([fwhm]),
        "z_vals": z_vals
    }, fmt)

# ────────────────────────────── CLI entry - point ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Digital Twin → FMC → TFM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Physical parameters ---------------------------------------------------
    parser.add_argument('--lx',  type=safe_float, required=True, help='Element width [mm]')
    parser.add_argument('--ly',  type=safe_float, required=True, help='Element height [mm]')
    parser.add_argument('--gx',  type=safe_float, required=True, help='Kerf (x) [mm]')
    parser.add_argument('--gy',  type=safe_float, required=True, help='Kerf (y) [mm]')
    parser.add_argument('--f',   type=safe_float, required=True, help='Centre freq. [MHz]')
    parser.add_argument('--d1',  type=safe_float, required=True, help='Water path [mm]')
    parser.add_argument('--c1',  type=safe_float, required=True, help='Water speed [m/s]')
    parser.add_argument('--d2',  type=safe_float, required=True, help='Steel path [mm]')
    parser.add_argument('--c2',  type=safe_float, required=True, help='Steel Cp speed [m/s]')
    parser.add_argument('--cs2', type=safe_float, required=True, help='Steel Cs speed [m/s]')

    # Array configuration ---------------------------------------------------
    parser.add_argument('--L1', type=int, default=11, help='Elements in X')
    parser.add_argument('--L2', type=int, default=11, help='Elements in Y')
    parser.add_argument('--ampx_type', choices=['rect','cos','Han','Ham','Blk','tri'],
                        default='rect', help='X-apodisation window')
    parser.add_argument('--ampy_type', choices=['rect','cos','Han','Ham','Blk','tri'],
                        default='rect', help='Y-apodisation window')
    parser.add_argument('--b', type=safe_float, default=0.15,
                        help='Element elev. aperture [mm]')

    # Beam steering / focusing ----------------------------------------------
    parser.add_argument('--angt',    type=safe_float, default=0.0, help='Array tilt [deg]')
    parser.add_argument('--theta20', type=safe_float, default=20.0, help='Refracted angle [deg]')
    parser.add_argument('--phi',     type=safe_float, default=0.0, help='Out-of-plane angle [deg]')
    parser.add_argument('--DF',      type=safe_float, default=float('inf'), help='F-number')
    parser.add_argument('--Dt0',     type=safe_float, default=50.8, help='Interface depth [mm]')
    parser.add_argument('--wave_type', choices=['p','s'], default='p', help='Wave type in steel')

    # Scan parameters --------------------------------------------------------
    parser.add_argument('--xs',    type=str, default='-5,20,100', help='X-scan spec')
    parser.add_argument('--zs',    type=str, default='1,20,100',  help='Z-scan spec')
    parser.add_argument('--y_vec', type=str, default='0',         help='Y position(s)')
    parser.add_argument('--z_mm',  type=str, default=None,        help='TFM depth vector')

    # Output control ---------------------------------------------------------
    parser.add_argument('--out_root', type=str, default='results', help='Output root dir')
    parser.add_argument('--save_fmt', choices=['csv','npz'], default='csv', help='Save format')

    args = parser.parse_args()
    ensure_dir(args.out_root)
    save_run_params(args, args.out_root)

    run_digital_twin_field(args, args.out_root, args.save_fmt)
    if args.z_mm:
        run_fmc_tfm(args, args.z_mm, args.out_root, args.save_fmt)

    print("[AP] Pipeline execution complete")

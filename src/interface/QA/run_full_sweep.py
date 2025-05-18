#!/usr/bin/env python3
"""
run_full_sweep.py (Optimized)

Perform a 2‑D parameter sweep over steering angles (phis), f‑numbers (fnums), and depths,
computing FMC via memoized element‑wise caching, running TFM back‑projection, and
saving quantitative metrics for each case.
"""
import os
import argparse
import json
import numpy as np
from multiprocessing import Pool
from scipy.signal import hilbert


# add project src to path if you're running from project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.rs_2Dv_service import run_rs_2Dv_service

# low‑level servicesrom application.rs_2Dv_service import run_rs_2Dv_service
from application.delay_laws2D_service import run_delay_laws2D_service

# higher‑level imaging routines
from interface.pipeline_tfm_backprojection_demo import tfm_backprojection

def compute_fmc_memoized(M, pitch, b, f, c, z_mm):
    elem_pos = (np.arange(M) - (M - 1) / 2) * pitch
    tof_us   = 2 * z_mm * 1e-3 / c * 1e6
    FMC      = np.zeros((M, M, z_mm.size), dtype=float)

    # 1) cache one‐way A‐scans
    p_cache = {}
    for idx, e in enumerate(elem_pos):
        p_cache[idx] = np.real(run_rs_2Dv_service(b, f, c, e, 0.0, z_mm))

    # 2) build the FMC cube by summing
    for tx in range(M):
        for rx in range(M):
            FMC[tx, rx, :] = p_cache[tx] + p_cache[rx]

    return tof_us, FMC

def compute_metrics(img, x_im, z_im):
    """
    Dummy stub: replace with your FWHM/CNR extraction code.
    Returns a dict of metrics for this image.
    """
    return {
        'max_amp': float(img.max()),
        'min_amp': float(img.min()),
        'mean_amp': float(img.mean()),
    }

def run_case(params):
    phi, fnum, z0, args = params
    # unpack common parameters
    M, pitch, b, f0, c = args.M, args.pitch, args.b, args.f0, args.c
    dx, dz = args.dx, args.dz

    # generate depth sample centered at z0
    Nz = int((args.z_max - args.z_min) / dz) + 1
    z_mm = np.linspace(args.z_min, args.z_max, Nz)

    # 1) FMC (memoized)
    tof_us, FMC = compute_fmc_memoized(M, pitch, b, f0, c, z_mm)

    # 2) imaging grid
    Nx = int((args.x_max - args.x_min) / dx) + 1
    Nz_im = Nz  # or override if you want coarser
    x_im = np.linspace(args.x_min, args.x_max, Nx)
    z_im = z_mm.copy()

    # 3) TFM backprojection
    img = tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c)

    # 4) metrics
    metrics = compute_metrics(img, x_im, z_im)

    # 5) package up results
    out = {
        'phi': phi,
        'fnum': fnum,
        'depth': z0,
        'metrics': metrics
    }
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phis',   type=float, nargs='+', required=True,
                   help='Steering angles (deg).')
    p.add_argument('--fnums',  type=float, nargs='+', required=True,
                   help='F‑numbers (focus depths / aperture).')
    p.add_argument('--depths', type=float, nargs='+', required=True,
                   help='Scatterer depths (mm).')
    p.add_argument('--dx',     type=float, default=0.1, help='Image dx (mm).')
    p.add_argument('--dz',     type=float, default=0.1, help='Image dz (mm).')
    p.add_argument('--output', type=str,   required=True,
                   help='Directory to save results.')
    p.add_argument('--jobs',   type=int,   default=1,
                   help='Number of parallel workers.')
    # low‑level array / medium defaults
    p.add_argument('--M',      type=int,   default=64,   help='# of elements')
    p.add_argument('--pitch',  type=float, default=0.5,  help='Element pitch (mm)')
    p.add_argument('--b',      type=float, default=0.25, help='Half‑element width (mm)')
    p.add_argument('--f0',     type=float, default=5.0,  help='Center frequency (MHz)')
    p.add_argument('--c',      type=float, default=1500, help='Sound speed (m/s)')
    p.add_argument('--z_min',  type=float, default=5.0,  help='Min depth (mm)')
    p.add_argument('--z_max',  type=float, default=60.0, help='Max depth (mm)')
    p.add_argument('--x_min',  type=float, default=-8.0, help='Min lateral (mm)')
    p.add_argument('--x_max',  type=float, default= 8.0, help='Max lateral (mm)')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # build sweep list
    sweep = []
    for phi in args.phis:
        for fnum in args.fnums:
            for z0 in args.depths:
                sweep.append((phi, fnum, z0, args))

    # run in parallel
    with Pool(args.jobs) as pool:
        results = pool.map(run_case, sweep)

    # save JSON
    out_file = os.path.join(args.output, 'sweep_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} cases → {out_file}")

if __name__ == '__main__':
    main()

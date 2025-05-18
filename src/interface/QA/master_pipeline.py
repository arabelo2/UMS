#!/usr/bin/env python3
# src/interface/master_pipeline.py 

"""
master_pipeline.py - Complete Digital Twin → FMC → TFM workflow
"""

import sys
import os
import numpy as np
import argparse
from scipy.signal import hilbert

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.delay_laws3Dint_service import run_delay_laws3Dint_service
from application.rs_2Dv_service import run_rs_2Dv_service
from application.mps_array_model_int_service import run_mps_array_model_int_service
from interface.cli_utils import safe_float

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_scan_vector(input_val, default_start, default_stop, default_num):
    if input_val is None:
        return np.linspace(default_start, default_stop, default_num)
    if not isinstance(input_val, str):
        return np.array(input_val, dtype=float)
    
    s = input_val.strip()
    if ':' in s:
        parts = [float(x) for x in s.split(':')]
        return np.arange(parts[0], parts[2] + parts[1], parts[1])
    parts = [p for p in s.split(',') if p]
    nums = [float(p) for p in parts]
    return np.linspace(nums[0], nums[1], int(nums[2])) if len(nums) == 3 else np.array(nums)

def save_data(folder, name, arrays, fmt):
    if fmt == 'npz':
        np.savez(os.path.join(folder, f"{name}.npz"), **arrays)
    else:
        for key, arr in arrays.items():
            path = os.path.join(folder, f"{name}_{key}.csv")
            if np.ndim(arr) == 3:
                for i in range(arr.shape[2]):
                    np.savetxt(path.replace('.csv', f'_z{i}.csv'), 
                              arr[:,:,i], delimiter=',')
            else:
                np.savetxt(path, np.atleast_2d(arr), delimiter=',')
    print(f"Saved data to {folder}/{name}.*")

def calculate_envelope_fwhm(signal, z_vals):
    signal_mag = np.abs(signal) if np.iscomplexobj(signal) else signal
    analytic_signal = hilbert(signal_mag)
    envelope = np.abs(analytic_signal)
    
    peak_idx = np.argmax(envelope)
    half_max = envelope[peak_idx] / 2
    
    above = envelope > half_max
    crossings = np.where(np.diff(above))[0]
    
    if len(crossings) >= 2:
        def interpolate_crossing(i):
            x = [z_vals[i], z_vals[i+1]]
            y = [envelope[i] - half_max, envelope[i+1] - half_max]
            return x[0] - y[0] * (x[1] - x[0]) / (y[1] - y[0])
        return envelope, interpolate_crossing(crossings[-1]) - interpolate_crossing(crossings[0])
    return envelope, 0.0

def run_digital_twin_field(params, out_root, fmt):
    out_dir = os.path.join(out_root, 'digital_twin')
    ensure_dir(out_dir)

    xs = parse_scan_vector(params.xs, -5, 20, 100)
    zs = parse_scan_vector(params.zs, 1, 20, 100)
    ys = parse_scan_vector(params.y_vec, 0, 0, 1)

    result = run_mps_array_model_int_service(
        params.lx, params.ly, params.gx, params.gy,
        params.f, params.d1, params.c1,
        params.d2, params.c2, params.cs2,
        params.wave_type, params.L1, params.L2,
        params.angt, params.Dt0, params.theta20,
        params.phi, params.DF,
        params.ampx_type,
        params.ampy_type,
        xs, zs, ys
    )

    save_data(out_dir, 'field', {
        'p_field': np.abs(result['p']),
        'x_vals': xs,
        'y_vals': ys,
        'z_vals': zs
    }, fmt)

def run_fmc_tfm(params, z_mm, out_root, fmt):
    out_dir = os.path.join(out_root, 'fmc_tfm')
    ensure_dir(out_dir)

    td = run_delay_laws3Dint_service(
        params.L1, params.L2,
        params.lx + params.gx, params.ly + params.gy,
        params.angt, params.phi,
        params.theta20, params.Dt0, params.DF,
        params.c1, params.c2,
        'n'
    )

    z_vals = parse_scan_vector(z_mm, 1, 20, 100)
    M = params.L1
    N = params.L2
    x_elem = (np.arange(M) - (M-1)/2) * (params.lx + params.gx)
    y_elem = (np.arange(N) - (N-1)/2) * (params.ly + params.gy)
    
    FMC = np.zeros((M, N, len(z_vals)), dtype=complex)
    
    for tx in range(M):
        for rx in range(N):
            FMC[tx, rx, :] = run_rs_2Dv_service(
                params.b, params.f, params.c1,
                x_elem[tx], y_elem[rx], z_vals
            )

    tfm_raw = np.zeros(len(z_vals), dtype=complex)
    sample_idx = np.arange(len(z_vals))
    
    for tx in range(M):
        for rx in range(N):
            # Handle all possible delay array shapes
            if td.ndim == 3:
                delays = td[tx, rx, :]
            elif td.ndim == 2:
                delays = np.full(len(z_vals), td[tx, rx])
            elif td.ndim == 1:
                if M == 1:  # Single transmitter
                    delays = np.full(len(z_vals), td[rx])
                elif N == 1:  # Single receiver
                    delays = np.full(len(z_vals), td[tx])
                else:  # M == N
                    delays = np.full(len(z_vals), td[tx])
            
            shifted_real = np.interp(
                sample_idx + delays,
                sample_idx,
                FMC[tx, rx, :].real,
                left=0, right=0
            )
            shifted_imag = np.interp(
                sample_idx + delays,
                sample_idx,
                FMC[tx, rx, :].imag,
                left=0, right=0
            )
            tfm_raw += shifted_real + 1j * shifted_imag

    envelope, fwhm = calculate_envelope_fwhm(np.abs(tfm_raw), z_vals)
    
    save_data(out_dir, 'results', {
        'delays': td,
        'fmc_real': FMC.real,
        'fmc_imag': FMC.imag,
        'tfm_raw_real': tfm_raw.real,
        'tfm_raw_imag': tfm_raw.imag,
        'envelope': envelope,
        'fwhm': np.array([fwhm]),
        'z_vals': z_vals
    }, fmt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Digital Twin to TFM Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Physical parameters
    parser.add_argument('--lx', '--element_width', type=safe_float, required=True, help='Element width [mm]')
    parser.add_argument('--ly', '--element_height', type=safe_float, required=True, help='Element height [mm]')
    parser.add_argument('--gx', '--kerf_x', type=safe_float, required=True, help='X-kerf width [mm]')
    parser.add_argument('--gy', '--kerf_y', type=safe_float, required=True, help='Y-kerf width [mm]')
    parser.add_argument('--f', '--frequency', type=safe_float, required=True, help='Center frequency [MHz]')
    parser.add_argument('--d1', '--water_thickness', type=safe_float, required=True, help='Water path [mm]')
    parser.add_argument('--c1', '--water_speed', type=safe_float, required=True, help='Water sound speed [m/s]')
    parser.add_argument('--d2', '--steel_thickness', type=safe_float, required=True, help='Steel thickness [mm]')
    parser.add_argument('--c2', '--steel_speed', type=safe_float, required=True, help='Steel longitudinal speed [m/s]')
    parser.add_argument('--cs2', '--steel_shear_speed', type=safe_float, required=True, help='Steel shear speed [m/s]')

    # Array configuration
    parser.add_argument('--L1', '--elements_x', type=int, default=11, help='Number of elements in X')
    parser.add_argument('--L2', '--elements_y', type=int, default=11, help='Number of elements in Y')
    parser.add_argument('--ampx_type', choices=['rect','cos','Han','Ham','Blk','tri'], default='rect', help='X-apodization window type')
    parser.add_argument('--ampy_type', choices=['rect','cos','Han','Ham','Blk','tri'], default='rect', help='Y-apodization window type')
    parser.add_argument('--b', '--element_aperture', type=safe_float, default=0.15, help='Element elevation aperture [mm]')

    # Beam steering
    parser.add_argument('--angt', '--array_tilt', type=safe_float, default=0.0, help='Array tilt angle [deg]')
    parser.add_argument('--theta20', '--refracted_angle', type=safe_float, default=20.0, help='Refracted angle in steel [deg]')
    parser.add_argument('--phi', '--out_of_plane_angle', type=safe_float, default=0.0, help='Out-of-plane angle [deg]')
    parser.add_argument('--DF', '--f_number', type=safe_float, default=float('inf'), help='F-number for focusing')
    parser.add_argument('--Dt0', '--interface_depth', type=safe_float, default=50.8, help='Interface depth [mm]')
    parser.add_argument('--wave_type', choices=['p','s'], default='p', help='Wave type in steel')

    # Scan parameters
    parser.add_argument('--xs', type=str, default='-5,20,100', help='X-scan vector specification')
    parser.add_argument('--zs', type=str, default='1,20,100', help='Z-scan vector specification')
    parser.add_argument('--y_vec', type=str, default='0', help='Y-position(s)')
    parser.add_argument('--z_mm', type=str, default=None, help='TFM depth vector')

    # Output control
    parser.add_argument('--out_root', type=str, default='results', help='Output directory root')
    parser.add_argument('--save_fmt', choices=['csv','npz'], default='csv', help='Output file format')

    args = parser.parse_args()
    ensure_dir(args.out_root)

    # Execute pipeline
    run_digital_twin_field(args, args.out_root, args.save_fmt)
    if args.z_mm:
        run_fmc_tfm(args, args.z_mm, args.out_root, args.save_fmt)

    print("[AP] Pipeline execution complete")
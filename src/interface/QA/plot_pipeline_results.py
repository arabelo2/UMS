#!/usr/bin/env python3
# src/interface/plot_pipeline_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_scan_vector(input_val, default_start, default_stop, default_num):
    """Parse scan vector specification (copied from master_pipeline)"""
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

def main():
    # -- Load Parameters --
    param_file = os.path.join("results", "run_params.json")
    try:
        with open(param_file) as f:
            params = json.load(f)
        print("Loaded parameters from run_params.json")
        
        # Calculate theoretical FWHM using loaded params
        theoretical_fwhm = (params['c2'] * 1000 / (params['f'] * 1e6)) * params['DF']
        
        # Get actual scan ranges
        x_scan = parse_scan_vector(params['xs'], -5, 20, 100)
        z_scan = parse_scan_vector(params['zs'], 1, 20, 100)
        focal_depth = z_scan[len(z_scan)//2]  # Midpoint of scan range
        
    except Exception as e:
        print(f"Warning: Could not load parameters ({str(e)}), using defaults")
        params = {
            'f': 5.0,  # MHz
            'c2': 5900, # m/s
            'DF': 45,   # F-number
            'L1': 11,   # elements x
            'L2': 11    # elements y
        }
        theoretical_fwhm = (params['c2'] * 1000 / (params['f'] * 1e6)) * params['DF']
        x_scan = np.linspace(-25, 25, 100)
        z_scan = np.linspace(0, 60, 100)
        focal_depth = 45  # mm
    
    # -- Configuration --
    plt.style.use('seaborn-v0_8-poster')
    plt.rcParams.update({
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # -- Directory Setup --
    dt_dir = os.path.join("results", "digital_twin")
    ft_dir = os.path.join("results", "fmc_tfm")
    ensure_dir("plots")

    # ------------------------------------------------------------------
    # 1. Digital Twin Field (|p|)
    # ------------------------------------------------------------------
    print("Processing Digital Twin Field...")
    try:
        x_vals = np.loadtxt(os.path.join(dt_dir, "field_x_vals.csv"), delimiter=',')
        z_vals = np.loadtxt(os.path.join(dt_dir, "field_z_vals.csv"), delimiter=',')
        p_field = np.loadtxt(os.path.join(dt_dir, "field_p_field.csv"), delimiter=',')
        
        p_field_norm = p_field / np.max(p_field)
        p_field_db = 20 * np.log10(p_field_norm + 1e-6)

        X, Z = np.meshgrid(x_vals, z_vals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        im1 = ax1.pcolormesh(X, Z, p_field_db, 
                           cmap='jet', 
                           shading='auto',
                           vmin=-40, vmax=0)
        ax1.set(xlabel="x (mm)", ylabel="z (mm)", 
               title="Digital Twin Field (dB)",
               aspect='equal',
               xlim=(x_scan.min(), x_scan.max()),
               ylim=(z_scan.min(), z_scan.max()))
        fig.colorbar(im1, ax=ax1, label="Amplitude (dB)")

        zoom_width = theoretical_fwhm * 3
        im2 = ax2.pcolormesh(X, Z, p_field_db,
                           cmap='jet',
                           shading='auto',
                           vmin=-20, vmax=0)
        ax2.set(xlabel="x (mm)", title=f"Zoomed View (-20dB cutoff, {zoom_width:.1f}mm width)",
               aspect='equal',
               xlim=(-zoom_width/2, zoom_width/2),
               ylim=(focal_depth - zoom_width/2, focal_depth + zoom_width/2))
        fig.colorbar(im2, ax=ax2, label="Amplitude (dB)")
        plt.tight_layout()
        plt.savefig("plots/digital_twin_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        focal_idx = np.argmin(np.abs(z_vals - focal_depth))
        plt.plot(x_vals, p_field_db[focal_idx, :], lw=2)
        plt.title(f"Beam Profile at z={focal_depth:.1f}mm")
        plt.xlabel("x (mm)"); plt.ylabel("Amplitude (dB)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/beam_profile.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error processing digital twin field: {str(e)}")

    # ------------------------------------------------------------------
    # 2. Delay Laws Analysis
    # ------------------------------------------------------------------
    print("\nProcessing Delay Laws...")
    try:
        delays = np.loadtxt(os.path.join(ft_dir, "results_delays.csv"), delimiter=',')
        
        if delays.ndim == 1:
            delays = delays.reshape(-1, 1)

        print(f"Delay Statistics:")
        print(f"Max: {delays.max():.4f} samples")
        print(f"Min: {delays.min():.4f} samples") 
        print(f"Range: {delays.max()-delays.min():.4f} samples")
        print(f"Sampling Check: {'OK' if delays.max() < 0.25 else 'WARNING'} (should be <0.25 samples)")

        plt.figure(figsize=(10, 8))
        im = plt.imshow(delays, aspect='auto', cmap='viridis', origin='lower')
        plt.xlabel("Rx Element Index")
        plt.ylabel("Tx Element Index")
        plt.title(f"Delay Laws (Array: {params['L1']}x{params['L2']})")
        cb = plt.colorbar(im, label="Delay (samples)")
        plt.xticks(np.arange(0, delays.shape[1], max(1, delays.shape[1]//8)))
        plt.yticks(np.arange(0, delays.shape[0], max(1, delays.shape[0]//8)))
        plt.tight_layout()
        plt.savefig("plots/delay_laws_heatmap.png", dpi=300)
        plt.close()

        center_tx = delays.shape[0] // 2
        tx_delay = delays[center_tx, :] if delays.shape[1] > 1 else delays.flatten()
        
        plt.figure(figsize=(12, 6))
        markerline, stemlines, _ = plt.stem(tx_delay, 
                                          linefmt='C0-',
                                          markerfmt='C0o',
                                          basefmt=' ')
        plt.setp(stemlines, 'linewidth', 1)
        
        thresh = 0.06
        plt.axhline(thresh, color='gray', ls='--', label='Max Allowable Delay')
        if delays.shape[1] > 1:
            plt.axvline(center_tx, color='gray', ls=':', alpha=0.5, label='Array Center')
        
        plt.xlabel("Element Index")
        plt.ylabel("Delay (samples)")
        plt.title(f"Delay Profile for {'Center Tx' if delays.shape[1]>1 else ''} Element {center_tx}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/delay_laws_stem.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error processing delay laws: {str(e)}")

    # ------------------------------------------------------------------
    # 3. TFM Envelope Analysis
    # ------------------------------------------------------------------
    print("\nProcessing TFM Envelope...")
    try:
        envelope = np.loadtxt(os.path.join(ft_dir, "results_envelope.csv"), delimiter=',')
        z_tfm = np.loadtxt(os.path.join(ft_dir, "results_z_vals.csv"), delimiter=',')

        envelope = envelope / np.max(envelope)
        envelope[envelope < 1e-6] = 1e-6

        peak_val = np.max(envelope)
        half_max = 0.5 * peak_val
        print(f"Envelope peak: {peak_val:.6f}, Half-max threshold: {half_max:.6f}")

        # Find the index of the peak (focal point)
        peak_idx = np.argmax(envelope)
        # Find where the envelope crosses the half-max
        above = envelope >= half_max
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]
        print(f"Half-max crossings found: {len(crossings)}")

        fwhm = np.nan
        z_left = z_right = np.nan

        if len(crossings) >= 2:
            # Boundary checks for interpolation
            if crossings[0] > 0:
                z_left_slice = z_tfm[crossings[0]-1:crossings[0]+2]
                env_left_slice = envelope[crossings[0]-1:crossings[0]+2]
            else:
                z_left_slice = z_tfm[0:3]
                env_left_slice = envelope[0:3]

            if crossings[-1] < len(z_tfm) - 1:
                z_right_slice = z_tfm[crossings[-1]-1:crossings[-1]+2]
                env_right_slice = envelope[crossings[-1]-1:crossings[-1]+2]
            else:
                z_right_slice = z_tfm[-3:]
                env_right_slice = envelope[-3:]

            def quad_interp(x, y, y0):
                coeffs = np.polyfit(x, y - y0, 2)
                roots = np.roots(coeffs)
                return roots[np.isreal(roots)].real[0] + x[0]

            try:
                z_left = quad_interp(z_left_slice, env_left_slice, half_max)
                z_right = quad_interp(z_right_slice, env_right_slice, half_max)
                fwhm = max(0, z_right - z_left)
            except Exception as e:
                print(f"Warning: Failed to interpolate FWHM: {e}")
                fwhm = np.nan
        else:
            print("Warning: Could not find sufficient half-max crossings to calculate FWHM.")

        print(f"FWHM Analysis:")
        print(f"Theoretical: {theoretical_fwhm:.2f} mm")
        if np.isnan(fwhm):
            print("Measured: N/A")
            print("Relative Error: N/A")
        else:
            print(f"Measured: {fwhm:.2f} mm")
            print(f"Relative Error: {abs(fwhm - theoretical_fwhm) / theoretical_fwhm * 100:.1f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(z_tfm, envelope, lw=2, label='Envelope')
        plt.axhline(half_max, ls='--', color='gray', label='Half Max')

        if not np.isnan(fwhm):
            plt.axvline(z_left, ls=':', color='red', alpha=0.7)
            plt.axvline(z_right, ls=':', color='red', alpha=0.7)
            plt.fill_betweenx([0, 1], z_left, z_right,
                              color='red', alpha=0.1,
                              label=f'FWHM={fwhm:.2f}mm')

        plt.xlabel("Depth z (mm)")
        plt.ylabel("Normalized Amplitude")
        plt.title(f"TFM Envelope (Theory FWHM={theoretical_fwhm:.2f}mm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/tfm_envelope.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(z_tfm, envelope, label='Raw Envelope')
        if not np.isnan(fwhm):
            plt.axvspan(z_left, z_right, color='red', alpha=0.1)
        plt.axhline(half_max, ls='--', color='gray')
        plt.title("Envelope Diagnostic View")
        plt.xlabel("Depth (mm)"); plt.ylabel("Amplitude")
        plt.legend()
        plt.savefig("plots/envelope_diagnostic.png")
        plt.close()

    except Exception as e:
        print(f"Error processing TFM envelope: {str(e)}")

    print("\nGenerated plots in 'plots' directory:")
    print(" • digital_twin_field.png - Field patterns with zoomed view")
    print(" • beam_profile.png - Cross-section at focal depth") 
    print(" • delay_laws_heatmap.png - Full delay matrix")
    print(" • delay_laws_stem.png - Delay profile with thresholds")
    print(" • tfm_envelope.png - Focused beam with FWHM markers")
    print(" • envelope_diagnostic.png - Raw envelope view")

if __name__ == "__main__":
    main()
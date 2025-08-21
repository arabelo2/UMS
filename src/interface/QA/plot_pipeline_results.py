#!/usr/bin/env python3
# src/interface/plot_pipeline_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json
from fwhm_methods_addon import estimate_all_fwhm_methods
from scipy.ndimage import gaussian_filter1d

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
    
        # Debug info inside try
        print("[DEBUG] Parameters loaded from run_params.json:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
        if 'DF' not in params or params['DF'] is None:
            raise ValueError("DF parameter is missing in run_params.json!")
    
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
        'font.size': 24,               # default font
        'axes.titlesize': 24,          # title font
        'axes.labelsize': 24,          # x and y labels
        'xtick.labelsize': 24,         # x-tick labels
        'ytick.labelsize': 24,         # y-tick labels
        'legend.fontsize': 24,         # legend text
        'figure.titlesize': 24,        # figure title
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

        # --- Full Field Plot (Left)
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

        # --- Adaptive -20 dB Region for Zoomed Plot (Right)
        focal_idx_z = np.argmin(np.abs(z_vals - focal_depth))
        x_profile = p_field_db[focal_idx_z, :]
        x_mask = x_profile > -20
        x_zoom_min, x_zoom_max = x_vals[x_mask].min(), x_vals[x_mask].max()

        focal_idx_x = np.argmax(np.max(p_field_db, axis=0))
        z_profile = p_field_db[:, focal_idx_x]
        z_mask = z_profile > -20
        z_zoom_min, z_zoom_max = z_vals[z_mask].min(), z_vals[z_mask].max()

        im2 = ax2.pcolormesh(X, Z, p_field_db,
                             cmap='jet',
                             shading='auto',
                             vmin=-20, vmax=0)
        ax2.set(xlabel="x (mm)",
                title=f"Zoomed View (-20dB cutoff)",
                aspect='equal',
                xlim=(x_zoom_min, x_zoom_max),
                ylim=(z_zoom_min, z_zoom_max))
        fig.colorbar(im2, ax=ax2, label="Amplitude (dB)")

        plt.tight_layout()
        plt.savefig("plots/digital_twin_field.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- Beam Profile at Focal Depth
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, x_profile, lw=2)
        plt.title(f"Beam Profile at z={focal_depth:.1f}mm")
        plt.xlabel("x (mm)")
        plt.ylabel("Amplitude (dB)")
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

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(delays, aspect='auto', cmap='viridis', origin='lower')
        plt.xlabel("Rx Element Index")
        plt.ylabel("Tx Element Index")
        plt.title(f"Delay Laws (Array: {params['L1']}x{params['L2']})")
        cb = plt.colorbar(im, label="Delay (samples)")

        # Dynamically determine ticks (max 8 ticks)
        max_ticks = 8
        xtick_step = max(1, delays.shape[1] // max_ticks)
        ytick_step = max(1, delays.shape[0] // max_ticks)
        xticks = np.arange(0, delays.shape[1], xtick_step)
        yticks = np.arange(0, delays.shape[0], ytick_step)     
        xtick_labels = [f"Rx{i}" for i in xticks]
        ytick_labels = [f"Tx{i}" for i in yticks]
        plt.xticks(xticks, xtick_labels, rotation=45)
        plt.yticks(yticks, ytick_labels)        
        plt.tight_layout()
        plt.savefig("plots/delay_laws_heatmap.png", dpi=300)
        plt.close()

        # Plot 1D stem for center TX
        center_tx = delays.shape[0] // 2
        tx_delay = delays[center_tx, :] if delays.shape[1] > 1 else delays.flatten()

        plt.figure(figsize=(12, 6))
        markerline, stemlines, _ = plt.stem(tx_delay, 
                                            linefmt='C0-', markerfmt='C0o', basefmt=' ')
        plt.setp(stemlines, 'linewidth', 1)

        thresh = 0.25
        plt.axhline(thresh, color='gray', ls='--', label='Max Allowable Delay')
        if delays.shape[1] > 1:
            plt.axvline(delays.shape[1] // 2, color='gray', ls=':', alpha=0.3, label='Array Center')

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

        # Normalize and smooth envelope (Rainio recommends smoothing for F1)
        envelope = envelope / np.max(envelope)
        dz = z_tfm[1] - z_tfm[0]  # axial resolution in mm
        print(f"[DEBUG] dz: {dz}")
        sigma_mm = 0.5 # Gaussian sigma in mm (Rainio suggests 0.3–1.0 mm typical)
        print(f"[DEBUG] sigma_mm: {sigma_mm}")
        sigma_samples = sigma_mm / dz
        print(f"[DEBUG] sigma: {sigma_samples}")
        envelope_smooth = gaussian_filter1d(envelope, sigma=sigma_samples)

        # Compute F1 from smoothed envelope
        peak_val = np.max(envelope_smooth)
        half_max = 0.5 * peak_val
        print(f"[DEBUG] Smoothed envelope: Peak={peak_val:.4f}, Half-max={half_max:.4f}")

        # Find where the envelope crosses the half-max
        above = envelope_smooth >= half_max
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]
        print(f"[DEBUG] Half-max crossings found: {len(crossings)}")
        print(f"[DEBUG] crossings: {crossings}")

        fwhm = np.nan
        if len(crossings) >= 2:
            # Define a local helper for robust interpolation
            def robust_interp(z_full, env_full, crossing_idx, target_val):
                # Ensure we get at least 2 points for linear interpolation,
                # ideally 3 for quadratic, respecting array bounds.
                idx_start = max(0, crossing_idx - 1)
                idx_end = min(len(env_full) - 1, crossing_idx + 1)
                
                # Make sure we have enough points for quadratic if possible
                # If only 2 points are available (e.g., at edge), extend to 3 if feasible
                if (idx_end - idx_start + 1) < 3:
                    if idx_start == 0 and len(env_full) >= 3: # Crossing at start, expand right
                        idx_end = 2
                    elif idx_end == len(env_full) - 1 and len(env_full) >= 3: # Crossing at end, expand left
                        idx_start = len(env_full) - 3
                    # If still less than 3 points, we'll fall back to linear
                
                x_slice = z_full[idx_start : idx_end + 1]
                y_slice = env_full[idx_start : idx_end + 1]

                if len(x_slice) < 2:
                    return np.nan # Not enough points for any interpolation
                elif len(x_slice) == 2:
                    # Fallback to linear interpolation if only 2 points
                    x1, x2 = x_slice
                    y1, y2 = y_slice
                    if y1 == y2: return np.nan # Avoid division by zero
                    return x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
                else: # len(x_slice) >= 3, attempt quadratic
                    try:
                        coeffs = np.polyfit(x_slice, y_slice - target_val, 2)
                        roots = np.roots(coeffs)
                        real_roots = roots[np.isreal(roots)].real
                        if len(real_roots) == 0:
                            # If no real roots, fall back to linear
                            x1, x2 = x_slice[0], x_slice[-1]
                            y1, y_slice[-1] = y_slice[0], y_slice[-1]
                            if y1 == y_slice[-1]: return np.nan
                            return x1 + (target_val - y1) * (x2 - x1) / (y_slice[-1] - y1)
                        # Choose the root closest to the actual crossing point (z_full[crossing_idx])
                        return real_roots[np.argmin(np.abs(real_roots - z_full[crossing_idx]))]
                    except np.linalg.LinAlgError:
                        # polyfit might fail for collinear points, fall back to linear
                        x1, x2 = x_slice[0], x_slice[-1]
                        y1, y2 = y_slice[0], y_slice[-1]
                        if y1 == y2: return np.nan
                        return x1 + (target_val - y1) * (x2 - x1) / (y2 - y1)
                    except Exception as e:
                        print(f"DEBUG: Error in robust_interp: {e}")
                        return np.nan

            try:
                z_left = robust_interp(z_tfm, envelope_smooth, crossings[0], half_max)
                z_right = robust_interp(z_tfm, envelope_smooth, crossings[-1], half_max)
                
                if not np.isnan(z_left) and not np.isnan(z_right):
                    fwhm = max(0, z_right - z_left)
                else:
                    print("Warning: One or both interpolated FWHM points are NaN.")

            except Exception as e:
                print(f"Warning: Failed to interpolate FWHM: {e}")
                fwhm = np.nan
            
        else:
            print("Warning: Could not find sufficient half-max crossings to calculate FWHM.")

        print(f"[DEBUG] FWHM: {fwhm}")
        
        # This block should be here, outside the if/else
        print(f"FWHM Analysis:")
        print(f"Theoretical: {theoretical_fwhm:.2f} mm")
        if np.isnan(fwhm):
            print("Measured: N/A")
            print("Relative Error: N/A")
        else:
            print(f"Measured: {fwhm:.2f} mm")
            print(f"Relative Error: {abs(fwhm - theoretical_fwhm) / theoretical_fwhm * 100:.1f}%")
        
        # Smooth envelope        
        envelope_smooth_for_plot = gaussian_filter1d(envelope, sigma=2.0) # Use this for plotting only to avoid re-smoothing
                                                                           # and affecting F1 if F1 used a different sigma
        # Extract ±20 mm window around peak        
        peak_idx = np.argmax(envelope_smooth_for_plot)        
        peak_z = z_tfm[peak_idx]
        mask = (z_tfm >= peak_z - 20) & (z_tfm <= peak_z + 20)
        z_clip = z_tfm[mask]
        env_clip = envelope_smooth_for_plot[mask]

        # Run all FWHM estimators on the clean window
        results = estimate_all_fwhm_methods(z_clip, env_clip, theoretical_fwhm, fwhm_f1=fwhm, save_csv=True, csv_path="plots/fwhm_comparison.csv", show_plot=True)
       
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
#!/usr/bin/env python3
# src/interface/plot_pipeline_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
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
        
        # Normalize and convert to dB
        p_field_norm = p_field / np.max(p_field)
        p_field_db = 20 * np.log10(p_field_norm + 1e-6)

        # Main field plot
        X, Z = np.meshgrid(x_vals, z_vals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full field view
        im1 = ax1.pcolormesh(X, Z, p_field_db, 
                           cmap='jet', 
                           shading='auto',
                           vmin=-40, vmax=0)
        ax1.set(xlabel="x (mm)", ylabel="z (mm)", 
               title="Digital Twin Field (dB)",
               aspect='equal',
               xlim=(-25, 25), ylim=(0, 60))
        fig.colorbar(im1, ax=ax1, label="Amplitude (dB)")

        # Zoomed view
        im2 = ax2.pcolormesh(X, Z, p_field_db,
                           cmap='jet',
                           shading='auto',
                           vmin=-20, vmax=0)
        ax2.set(xlabel="x (mm)", title="Zoomed View (-20dB cutoff)",
               aspect='equal',
               xlim=(-10, 10), ylim=(20, 40))
        fig.colorbar(im2, ax=ax2, label="Amplitude (dB)")
        plt.tight_layout()
        plt.savefig("plots/digital_twin_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional beam profile plot
        focal_depth = 45  # mm
        focal_idx = np.argmin(np.abs(z_vals - focal_depth))
        plt.figure(figsize=(10, 5))
        plt.plot(x_vals, p_field_db[focal_idx, :], lw=2)
        plt.title(f"Beam Profile at z={focal_depth}mm")
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
        
        # Reshape if needed
        if delays.ndim == 1:
            delays = delays.reshape(-1, 1)

        # Calculate statistics
        print(f"Delay Statistics:")
        print(f"Max: {delays.max():.4f} samples")
        print(f"Min: {delays.min():.4f} samples") 
        print(f"Range: {delays.max()-delays.min():.4f} samples")
        print(f"Sampling Check: {'OK' if delays.max() < 0.25 else 'WARNING'} (should be <0.25 samples)")

        # 2a) Heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(delays, aspect='auto', cmap='viridis', origin='lower')
        plt.xlabel("Rx Element Index")
        plt.ylabel("Tx Element Index")
        plt.title(f"Delay Laws (Array: {delays.shape[0]}x{delays.shape[1]})")
        cb = plt.colorbar(im, label="Delay (samples)")
        plt.xticks(np.arange(0, delays.shape[1], max(1, delays.shape[1]//8)))
        plt.yticks(np.arange(0, delays.shape[0], max(1, delays.shape[0]//8)))
        plt.tight_layout()
        plt.savefig("plots/delay_laws_heatmap.png", dpi=300)
        plt.close()

        # 2b) Stem Plot
        center_tx = delays.shape[0] // 2
        tx_delay = delays[center_tx, :] if delays.shape[1] > 1 else delays.flatten()
        
        plt.figure(figsize=(12, 6))
        markerline, stemlines, _ = plt.stem(tx_delay, 
                                          linefmt='C0-',
                                          markerfmt='C0o',
                                          basefmt=' ')
        plt.setp(stemlines, 'linewidth', 1)
        
        # Annotations
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

        # Normalize and ensure non-zero
        envelope = envelope / np.max(envelope)
        envelope[envelope < 1e-6] = 1e-6  # Prevent log(0)
        
        # Enhanced FWHM Calculation
        peak_idx = np.argmax(envelope)
        half_max = 0.5
        above = envelope > half_max
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]
        
        if len(crossings) >= 2:
            # Quadratic interpolation for better accuracy
            def quad_interp(x, y, y0):
                coeffs = np.polyfit(x, y - y0, 2)
                roots = np.roots(coeffs)
                return roots[np.isreal(roots)].real[0] + x[1]
            
            try:
                z_left = quad_interp(
                    z_tfm[crossings[0]-1:crossings[0]+2],
                    envelope[crossings[0]-1:crossings[0]+2],
                    half_max
                )
                z_right = quad_interp(
                    z_tfm[crossings[-1]-1:crossings[-1]+2],
                    envelope[crossings[-1]-1:crossings[-1]+2],
                    half_max
                )
                fwhm = max(0, z_right - z_left)  # Ensure non-negative
            except:
                z_left = z_right = fwhm = 0
        else:
            z_left = z_right = fwhm = 0

        # Theoretical FWHM
        c2 = 5900  # Steel longitudinal speed (m/s)
        f = 5e6    # Frequency (Hz)
        lambda_steel = c2 / f * 1000  # wavelength in mm
        theoretical_fwhm = lambda_steel * 45  # Using your F=45
        
        print(f"FWHM Analysis:")
        print(f"Theoretical: {theoretical_fwhm:.2f} mm")
        print(f"Measured: {fwhm:.2f} mm")
        print(f"Relative Error: {abs(fwhm-theoretical_fwhm)/theoretical_fwhm*100:.1f}%")

        # Plot with enhanced features
        plt.figure(figsize=(12, 6))
        plt.plot(z_tfm, envelope, lw=2, label='Envelope')
        plt.axhline(half_max, ls='--', color='gray', label='Half Max')
        
        if fwhm > 0:
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
        
        # Additional diagnostic plot
        plt.figure()
        plt.plot(z_tfm, envelope, label='Raw Envelope')
        if fwhm > 0:
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
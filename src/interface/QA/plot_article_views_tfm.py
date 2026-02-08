import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

def calculate_fwhm(axis, profile, level_db=-6):
    """Calculates Beam Width at specific dB level"""
    above_thresh = np.where(profile >= level_db)[0]
    if len(above_thresh) > 0:
        left_idx = above_thresh[0]
        right_idx = above_thresh[-1]
        width = axis[right_idx] - axis[left_idx]
        return width, axis[left_idx], axis[right_idx]
    return 0.0, 0.0, 0.0

def plot_publication_figure_tfm(root_dir):
    try:
        # 1. Load Data
        p_path = os.path.join(root_dir, 'results_envelope_2d.csv')
        x_path = os.path.join(root_dir, 'results_x_vals.csv')
        z_path = os.path.join(root_dir, 'results_z_vals.csv')

        if not os.path.exists(p_path):
            print(f"[ERROR] TFM files not found in {root_dir}")
            return

        p_raw = pd.read_csv(p_path, header=None).values
        x_vals = pd.read_csv(x_path, header=None).values.flatten()
        z_vals = pd.read_csv(z_path, header=None).values.flatten()
        
        # 2. Align Data Shape (Robust Transpose)
        if p_raw.shape == (len(z_vals), len(x_vals)):
            p_aligned = p_raw
        elif p_raw.shape == (len(x_vals), len(z_vals)):
            p_aligned = p_raw.T
        else:
            if p_raw.shape[0] == len(x_vals):
                p_aligned = p_raw.T
            else:
                p_aligned = p_raw

        # 3. dB Scale
        p_norm = np.abs(p_aligned)
        max_val = np.max(p_norm)
        if max_val > 0:
            p_norm = p_norm / max_val
            p_db = 20 * np.log10(p_norm + 1e-12)
            p_db = np.clip(p_db, -60, 0)
        else:
            p_db = np.zeros_like(p_norm) - 60
            
        p_db = np.flip(p_db, 1)

        # 4. Find Peak
        max_idx = np.unravel_index(np.argmax(p_db), p_db.shape)
        peak_z_idx, peak_x_idx = max_idx
        peak_z = z_vals[peak_z_idx]
        peak_x = x_vals[peak_x_idx]

        # 5. Extract Profiles
        profile_axial_db = p_db[:, peak_x_idx]
        profile_lateral_db = p_db[peak_z_idx, :]

        # 6. Metrics
        fwhm_lat, l_lat, r_lat = calculate_fwhm(x_vals, profile_lateral_db, -6)

        # 7. Plot
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # A: Map
        
        ax_map = fig.add_subplot(gs[:, 0])
        im = ax_map.pcolormesh(x_vals, z_vals, p_db, cmap='jet', shading='auto', vmin=-30, vmax=0)        
        ax_map.set_xlabel('Lateral Position (mm)')
        ax_map.set_ylabel('Depth (mm)')
        ax_map.set_title(f'TFM Reconstruction Field (dB)\nPeak at z={peak_z:.1f}mm')
        ax_map.axvline(peak_x, color='white', linestyle='--', alpha=0.5)
        ax_map.axhline(peak_z, color='white', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax_map, label='Normalized Amplitude (dB)')

        # B: Lateral
        ax_lat = fig.add_subplot(gs[0, 1])
        ax_lat.plot(x_vals, profile_lateral_db, 'b-', linewidth=2)
        ax_lat.set_title(f'Lateral Profile at z={peak_z:.1f}mm')
        ax_lat.set_ylabel('Amplitude (dB)')
        ax_lat.set_ylim(-40, 2)
        ax_lat.grid(True, linestyle='--', alpha=0.5)
        ax_lat.axhline(-6, color='r', linestyle=':', label='-6dB')
        ax_lat.axvline(l_lat, color='g', linestyle='--')
        ax_lat.axvline(r_lat, color='g', linestyle='--')
        ax_lat.text(0, -35, f"Width (-6dB):\n{fwhm_lat:.2f} mm", ha='center', 
                    bbox=dict(facecolor='white', alpha=0.8))

        # C: Axial
        ax_ax = fig.add_subplot(gs[1, 1])
        ax_ax.plot(z_vals, profile_axial_db, 'b-', linewidth=2)
        ax_ax.set_title(f'Axial Profile at x={peak_x:.1f}mm')
        ax_ax.set_xlabel('Depth (mm)')
        ax_ax.set_ylabel('Amplitude (dB)')
        ax_ax.set_ylim(-40, 2)
        ax_ax.grid(True, linestyle='--', alpha=0.5)
        ax_ax.axvline(peak_z, color='r', linestyle='-', label=f'Peak ({peak_z:.1f}mm)')
        ax_ax.legend()

        plt.tight_layout()
        save_path = os.path.join(root_dir, 'article_style_analysis_tfm.png')
        plt.savefig(save_path, dpi=300)
        print(f"[SUCCESS] Plot saved to: {save_path}")
        plt.close()

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    plot_publication_figure_tfm(target_dir)
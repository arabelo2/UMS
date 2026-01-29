import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import json
import sys

def load_data(root_dir):
    """Robustly loads and aligns simulation data"""
    try:
        # Load Raw Files
        p_raw = pd.read_csv(os.path.join(root_dir, 'field_p_field.csv'), header=None).values
        x_vals = pd.read_csv(os.path.join(root_dir, 'field_x_vals.csv'), header=None).values.flatten()
        z_vals = pd.read_csv(os.path.join(root_dir, 'field_z_vals.csv'), header=None).values.flatten()
        
        # Load Parameters for metadata
        params_path = os.path.join(root_dir, 'run_params.json')
        params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)

        # Smart Transpose Logic (Fixes the shape mismatch issue)
        # We expect Shape (Z_dim, X_dim)
        if p_raw.shape == (len(z_vals), len(x_vals)):
            p_aligned = p_raw
        elif p_raw.shape == (len(x_vals), len(z_vals)):
            p_aligned = p_raw.T
        else:
            # Fallback based on dimensions
            if p_raw.shape[0] == len(x_vals):
                p_aligned = p_raw.T
            else:
                p_aligned = p_raw
                
        # Calculate Amplitude and dB Scale
        # Normalize to Max = 0 dB
        amp = np.abs(p_aligned)
        max_val = np.max(amp)
        if max_val > 0:
            p_norm = amp / max_val
            # Clip small values to -60dB to avoid log(0)
            p_db = 20 * np.log10(p_norm + 1e-12)
            p_db = np.clip(p_db, -60, 0)
        else:
            p_norm = amp
            p_db = amp

        return x_vals, z_vals, p_norm, p_db, params

    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return None, None, None, None, None

def calculate_fwhm(axis, profile, level_db=-3):
    """Calculates Beam Width at specific dB level (e.g., -3dB or -6dB)"""
    # Profile should be in dB
    # Find indices where intensity > level_db
    above_thresh = np.where(profile >= level_db)[0]
    
    if len(above_thresh) > 0:
        left_idx = above_thresh[0]
        right_idx = above_thresh[-1]
        width = axis[right_idx] - axis[left_idx]
        return width, axis[left_idx], axis[right_idx]
    return 0.0, 0.0, 0.0

def plot_publication_figure(root_dir):
    x, z, p_norm, p_db, params = load_data(root_dir)
    if x is None: return

    # Find Peak
    max_idx = np.unravel_index(np.argmax(p_norm), p_norm.shape)
    peak_z_idx, peak_x_idx = max_idx
    peak_z = z[peak_z_idx]
    peak_x = x[peak_x_idx]

    # Extract Profiles
    profile_axial_db = p_db[:, peak_x_idx]
    profile_lateral_db = p_db[peak_z_idx, :]

    # Calculate Metrics
    fwhm_lat, l_lat, r_lat = calculate_fwhm(x, profile_lateral_db, -6) # -6dB width
    fwhm_ax, l_ax, r_ax = calculate_fwhm(z, profile_axial_db, -6)      # -6dB depth of field

    # --- PLOTTING SETUP ---
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Plot 1: Main 2D Map (dB Scale) [Top Left]
    ax_map = fig.add_subplot(gs[:, 0])
    im = ax_map.pcolormesh(x, z, p_db, cmap='jet', shading='auto', vmin=-30, vmax=0)
    ax_map.set_xlabel('Lateral Position (mm)')
    ax_map.set_ylabel('Depth (mm)')
    ax_map.set_title(f'Acoustic Pressure Field (dB)\nPeak at z={peak_z:.1f}mm')
    
    # Add crosshairs at peak
    ax_map.axvline(peak_x, color='white', linestyle='--', alpha=0.5)
    ax_map.axhline(peak_z, color='white', linestyle='--', alpha=0.5)
    
    # Add Interface line if exists
    if 'Dt0' in params:
        ax_map.axhline(params['Dt0'], color='cyan', linestyle='-', linewidth=1.5, label='Interface')
        ax_map.legend()

    plt.colorbar(im, ax=ax_map, label='Normalized Pressure (dB)')

    # Plot 2: Lateral Profile (Beam Width) [Top Right]
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_lat.plot(x, profile_lateral_db, 'b-', linewidth=2)
    ax_lat.set_title(f'Lateral Profile at z={peak_z:.1f}mm')
    ax_lat.set_ylabel('Amplitude (dB)')
    ax_lat.set_ylim(-40, 2)
    ax_lat.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Show Beam Width
    ax_lat.axhline(-6, color='r', linestyle=':', label='-6dB Level')
    ax_lat.axvline(l_lat, color='g', linestyle='--')
    ax_lat.axvline(r_lat, color='g', linestyle='--')
    ax_lat.text(0, -35, f"Beam Width (-6dB):\n{fwhm_lat:.2f} mm", ha='center', 
                bbox=dict(facecolor='white', alpha=0.8))

    # Plot 3: Axial Profile (Depth of Field) [Bottom Right]
    ax_ax = fig.add_subplot(gs[1, 1])
    ax_ax.plot(z, profile_axial_db, 'b-', linewidth=2)
    ax_ax.set_title(f'Axial Profile at x={peak_x:.1f}mm')
    ax_ax.set_xlabel('Depth (mm)')
    ax_ax.set_ylabel('Amplitude (dB)')
    ax_ax.set_ylim(-40, 2)
    ax_ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Show Target vs Actual
    if 'Dt0' in params and 'DF' in params:
        target_z = params['Dt0'] + params['DF']
        ax_ax.axvline(target_z, color='k', linestyle='--', label=f'Target ({target_z}mm)')
    
    ax_ax.axvline(peak_z, color='r', linestyle='-', label=f'Peak ({peak_z:.1f}mm)')
    ax_ax.legend()

    plt.tight_layout()
    
    save_path = os.path.join(root_dir, 'article_style_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Article-style figure saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Default to the path in your last command, or take arg
    default_path = "results/dissertation/test_45"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print(f"Processing directory: {target_dir}")
    plot_publication_figure(target_dir)
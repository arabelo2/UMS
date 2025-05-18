#!/usr/bin/env python3
"""
pipeline_metrics_sweep.py

Sweep resolution and contrast metrics over steering angles, focal distances, and depths.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def load_tfm_image(tfm_path, dx, dz):
    """
    Load TFM image from CSV and construct x, z arrays.
    """
    img = pd.read_csv(tfm_path, header=None).values.astype(float)
    n_z, n_x = img.shape
    x = np.arange(0, n_x * dx, dx)
    z = np.arange(0, n_z * dz, dz)
    return img, x, z


def compute_fwhm(profile, axis_spacing):
    """
    Compute Full-Width at Half-Maximum of a 1D profile.
    """
    peak = np.max(profile)
    half = peak / 2
    indices = np.where(profile >= half)[0]
    if indices.size < 2:
        return np.nan
    return (indices[-1] - indices[0]) * axis_spacing


def compute_metrics(img, x, z, x0, z0, window=5):
    """
    Given TFM image, compute lateral and axial FWHM and CNR at (x0, z0).
    """
    # find nearest indices
    ix = np.argmin(np.abs(x - x0))
    iz = np.argmin(np.abs(z - z0))

    # lateral profile at depth iz
    lat_prof = img[iz, :]
    lateral_fwhm = compute_fwhm(lat_prof, x[1] - x[0])

    # axial profile at lateral ix
    ax_prof = img[:, ix]
    axial_fwhm = compute_fwhm(ax_prof, z[1] - z[0])

    # contrast-to-noise ratio: signal is mean in small window around (ix,iz)
    win = window
    sig_patch = img[max(0, iz-win):min(len(z), iz+win+1),
                    max(0, ix-win):min(len(x), ix+win+1)]
    background = img.copy()
    # mask out signal region
    background[max(0, iz-win):min(len(z), iz+win+1),
               max(0, ix-win):min(len(x), ix+win+1)] = np.nan
    mu_s = np.nanmean(sig_patch)
    mu_b = np.nanmean(background)
    sigma_b = np.nanstd(background)
    cnr = (mu_s - mu_b) / (sigma_b + 1e-12)

    return lateral_fwhm, axial_fwhm, cnr


def main():
    parser = argparse.ArgumentParser(
        description="Sweep resolution/contrast metrics over parameters."
    )
    parser.add_argument("--tfm_file", required=True,
                        help="Path to TFM image CSV.")
    parser.add_argument("--dx", type=float, default=0.1,
                        help="Lateral pixel spacing (mm). Default: 0.1")
    parser.add_argument("--dz", type=float, default=0.1,
                        help="Axial pixel spacing (mm). Default: 0.1")
    parser.add_argument("--phis", type=float, nargs='+', default=[-20, -10, 0, 10, 20],
                        help="Steering angles (deg).")
    parser.add_argument("--Fs", type=float, nargs='+', default=[np.inf, 100.0, 50.0],
                        help="Focal distances (mm). Use inf for no focusing.")
    parser.add_argument("--z0s", type=float, nargs='+', default=[20.0, 30.0, 40.0, 50.0],
                        help="Depths for metric calc (mm).")
    parser.add_argument("--x0", type=float, default=0.0,
                        help="Lateral position for metrics (mm). Default: 0.0")
    parser.add_argument("--window", type=int, default=5,
                        help="Half-window size in pixels for signal region. Default: 5")
    parser.add_argument("--output_csv", default="metrics_sweep.csv",
                        help="Output CSV filename.")
    parser.add_argument("--output_plot", default="metrics_sweep.png",
                        help="Output plot filename.")
    args = parser.parse_args()

    # load image and coords
    img, x, z = load_tfm_image(args.tfm_file, args.dx, args.dz)

    records = []
    for Phi in args.phis:
        for F in args.Fs:
            for z0 in args.z0s:
                lat_fwhm, ax_fwhm, cnr = compute_metrics(
                    img, x, z, args.x0, z0, window=args.window
                )
                records.append({
                    'Phi_deg': Phi,
                    'F_mm': F,
                    'z0_mm': z0,
                    'lateral_FWHM_mm': lat_fwhm,
                    'axial_FWHM_mm': ax_fwhm,
                    'CNR': cnr
                })
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")

    # Plotting: lateral and axial FWHM vs depth
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    for Phi in args.phis:
        for label, metric in [('Lateral', 'lateral_FWHM_mm'), ('Axial', 'axial_FWHM_mm')]:
            ax = axs[0] if label=='Lateral' else axs[1]
            for F in args.Fs:
                subset = df[(df['Phi_deg']==Phi) & (df['F_mm']==F)]
                ax.plot(subset['z0_mm'], subset[metric], marker='o', label=f"Φ={Phi}°, F={ '∞' if np.isinf(F) else int(F) }")
            ax.set_ylabel(f"{label} FWHM (mm)")
            ax.grid(True)
    axs[0].set_title("Lateral FWHM vs Depth")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axs[1].set_title("Axial FWHM vs Depth")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # CNR plot
    ax3 = axs[2]
    for Phi in args.phis:
        for F in args.Fs:
            subset = df[(df['Phi_deg']==Phi) & (df['F_mm']==F)]
            ax3.plot(subset['z0_mm'], subset['CNR'], marker='o', label=f"Φ={Phi}°, F={ '∞' if np.isinf(F) else int(F) }")
    ax3.set_xlabel("Depth z0 (mm)")
    ax3.set_ylabel("CNR")
    ax3.set_title("Contrast-to-Noise Ratio vs Depth")
    ax3.grid(True)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    print(f"Plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()

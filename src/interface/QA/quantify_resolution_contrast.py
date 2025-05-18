#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
quantify_resolution_contrast.py

Compute resolution (FWHM) and contrast-to-noise ratio (CNR) for a TFM image stored as a CSV.
"""

import argparse
import numpy as np
import pandas as pd
import csv

def load_tfm_image(tfm_file: str, dx: float, dz: float):
    """
    Load a TFM image from a CSV (no header), return image array and coordinate vectors.
    """
    df = pd.read_csv(tfm_file, header=None)
    img = df.values.astype(float)
    nz, nx = img.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    return img, x, z

def find_nearest_index(arr: np.ndarray, val: float) -> int:
    """Return index of the element in arr closest to val."""
    return int(np.abs(arr - val).argmin())

def measure_fwhm(profile: np.ndarray, coords: np.ndarray) -> float:
    """
    Compute full-width at half-maximum (FWHM) of a 1D profile.
    Returns width in the same units as coords.
    """
    half_max = profile.max() / 2.0
    above = np.where(profile >= half_max)[0]
    if above.size < 2:
        return 0.0
    left, right = above[0], above[-1]
    return float(coords[right] - coords[left])

def compute_cnr(img: np.ndarray, x_idx: int, z_idx: int, roi_radius: int = 5) -> float:
    """
    Compute contrast‐to‐noise ratio around (z_idx, x_idx).
    Signal ROI is a (2*roi_radius+1)^2 patch; background is the rest.
    CNR = (μ_signal – μ_background) / σ_background.
    """
    # signal region
    z_min = max(z_idx - roi_radius, 0)
    z_max = min(z_idx + roi_radius + 1, img.shape[0])
    x_min = max(x_idx - roi_radius, 0)
    x_max = min(x_idx + roi_radius + 1, img.shape[1])
    signal = img[z_min:z_max, x_min:x_max].ravel()
    μ_sig = signal.mean()

    # background region
    mask = np.ones(img.shape, dtype=bool)
    mask[z_min:z_max, x_min:x_max] = False
    background = img[mask].ravel()
    μ_bkg = background.mean()
    σ_bkg = background.std(ddof=1)
    if σ_bkg == 0:
        return float('nan')
    return float((μ_sig - μ_bkg) / σ_bkg)

def main():
    parser = argparse.ArgumentParser(
        description="Quantify resolution (FWHM) and contrast (CNR) in a TFM image CSV."
    )
    parser.add_argument(
        "--tfm_file", required=True,
        help="Path to the input TFM image CSV (no header)."
    )
    parser.add_argument(
        "--dx", type=float, required=True,
        help="Lateral pixel spacing in mm."
    )
    parser.add_argument(
        "--dz", type=float, required=True,
        help="Axial pixel spacing in mm."
    )
    parser.add_argument(
        "--x0", type=float, required=True,
        help="Lateral coordinate of the point reflector (mm)."
    )
    parser.add_argument(
        "--z0", type=float, required=True,
        help="Axial coordinate of the point reflector (mm)."
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Path to save the metrics CSV."
    )
    args = parser.parse_args()

    # Load image and coordinates
    img, x, z = load_tfm_image(args.tfm_file, args.dx, args.dz)

    # Find nearest pixel to the reflector
    x_idx = find_nearest_index(x, args.x0)
    z_idx = find_nearest_index(z, args.z0)

    # Extract profiles at that point
    lateral_profile = img[z_idx, :]
    axial_profile   = img[:, x_idx]

    # Compute FWHM
    lateral_fwhm = measure_fwhm(lateral_profile, x)
    axial_fwhm   = measure_fwhm(axial_profile, z)

    # Compute CNR
    cnr = compute_cnr(img, x_idx, z_idx)

    # Assemble metrics
    metrics = {
        "x0_mm": args.x0,
        "z0_mm": args.z0,
        "lateral_FWHM_mm": lateral_fwhm,
        "axial_FWHM_mm": axial_fwhm,
        "CNR": cnr
    }

    # Write out CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

    print(f"Metrics saved to {args.output_csv}")
    print(metrics)

if __name__ == "__main__":
    main()

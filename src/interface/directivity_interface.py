#!/usr/bin/env python3
# src/interface/directivity_interface.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from interface.cli_utils import parse_array

def apply_plot_style(ax=None, title=None, xlabel=None, ylabel=None):
    if ax is None:
        ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=18)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

def find_closest_value(array, target):
    array = np.asarray(array)
    idx = np.searchsorted(array, target, side="left")
    if idx > 0 and (idx == len(array) or abs(target - array[idx-1]) <= abs(target - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def compute_directivity(v_slice, x, z_offset):
    theta = np.arctan2(x, z_offset) * (180.0 / np.pi)
    directivity = np.abs(v_slice)
    directivity /= np.max(directivity)
    return theta, directivity

def secondary_analysis(theta, directivity, snell_angle, tolerance=2.0):
    """
    Detect and quantify any weaker lobes near the Snell-predicted angle.
    """
    nearby_idx = np.where((theta >= snell_angle - tolerance) & (theta <= snell_angle + tolerance))[0]
    if len(nearby_idx) == 0:
        return None, None, None

    # Find peak within tolerance window
    peak_idx = nearby_idx[np.argmax(directivity[nearby_idx])]
    peak_amp = directivity[peak_idx]
    relative_dB = 20 * np.log10(np.clip(peak_amp, 1e-12, None))

    return theta[peak_idx], peak_amp, relative_dB

def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot directivity pattern with main & side lobes and Snell-angle analysis.",
        epilog=(
            "Example usage:\n"
            "  python src/interface/directivity_interface.py --infile velocity_output.txt \\\n"
            "      --x=\"-30,30,305\" --y=\"0,0,1\" --z=\"1,50,203\" \\\n"
            "      --freq 5.0 --z_offset 15 --mat \"1,1480,7.9,5900,3200,p\" \\\n"
            "      --snell_angle 22.5 --outfile \"directivity_output.txt\" --plotfile \"directivity.png\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--infile", type=str, required=True,
                        help="Path to velocity field matrix file.")
    parser.add_argument("--x", type=str, required=True,
                        help="Range or scalar for x (mm).")
    parser.add_argument("--y", type=str, required=True,
                        help="Range or scalar for y (mm).")
    parser.add_argument("--z", type=str, required=True,
                        help="Range or scalar for z (mm).")
    parser.add_argument("--z_offset", type=float, required=True,
                        help="Requested fixed z slice for angle calculation (mm).")
    parser.add_argument("--freq", type=float, required=True,
                        help="Operating frequency in MHz.")
    parser.add_argument("--mat", type=str, required=True,
                        help="Material parameters: d1,cp1,d2,cp2,cs2,wave (e.g., \"1,1480,7.9,5900,3200,p\").")
    parser.add_argument("--snell_angle", type=float, default=None,
                        help="Optional Snell-predicted angle to check for weaker lobes (degrees).")
    parser.add_argument("--outfile", type=str, default="directivity_output.txt",
                        help="Output file for directivity data.")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="Optional filename to save directivity plot.")
    parser.add_argument("--dB", action="store_true",
                        help="Plot directivity in dB scale.")

    args = parser.parse_args()

    # Parse grids
    x = parse_array(args.x)
    y = parse_array(args.y)
    z = parse_array(args.z)

    # Find closest z offset
    z_offset_actual = find_closest_value(z, args.z_offset)
    print(f"Requested z_offset: {args.z_offset} mm → Using closest z = {z_offset_actual:.6f} mm")

    # Parse material parameters: d1, cp1, d2, cp2, cs2, wave
    mat_params = args.mat.split(",")
    cp1 = float(mat_params[1])  # speed of sound in medium 1 (mm/s)

    # Compute Fraunhofer distance
    wavelength = cp1 / (args.freq * 1e6)  # Convert MHz to Hz
    aperture_D = max(x) - min(x)          # Aperture size (mm)
    z_far = (2 * (aperture_D ** 2)) / wavelength
    print(f"Fraunhofer distance z_far = {z_far:.3f} mm")

    # Decide near-field or far-field
    if z_offset_actual < z_far:
        field_region = "Near-Field Directivity"
    else:
        field_region = "Far-Field Directivity"

    # Load velocity field
    try:
        v = np.loadtxt(args.infile)
        v = v.reshape((len(z), len(y), len(x)))
    except Exception as e:
        sys.exit(f"Error loading or reshaping velocity data: {e}")

    # Slice at z_offset
    idx_z = np.argmin(np.abs(z - z_offset_actual))
    v_slice = v[idx_z, :, :]

    # Reduce to x-axis (if y is scalar)
    if len(y) == 1:
        v_slice = v_slice.flatten()
    else:
        v_slice = np.mean(v_slice, axis=0)

    # Compute directivity
    theta, directivity = compute_directivity(v_slice, x, z_offset_actual)

    # Find peaks
    peaks, _ = find_peaks(directivity, height=0.01)  # Ignore tiny peaks
    main_idx = np.argmax(directivity)
    side_lobe_idxs = [p for p in peaks if p != main_idx]

    # Check for Snell angle weaker lobe
    if args.snell_angle is not None:
        snell_theta, snell_amp, snell_dB = secondary_analysis(theta, directivity, args.snell_angle)
        if snell_theta is not None:
            print(f"Weaker lobe detected near Snell angle ({args.snell_angle}°):")
            print(f"  Angle: {snell_theta:.2f}°, Amplitude: {snell_amp:.3f}, Relative dB: {snell_dB:.2f} dB")
        else:
            print(f"No significant lobe detected near Snell angle ({args.snell_angle}°)")
    else:
        snell_theta = snell_amp = snell_dB = None

    # Save results
    try:
        with open(args.outfile, "w") as f:
            for i, (angle, amp) in enumerate(zip(theta, directivity)):
                lobe_type = "main" if i == main_idx else ("side" if i in side_lobe_idxs else "none")
                f.write(f"{angle:.6f}\t{amp:.16f}\t{lobe_type}\n")
        print(f"Directivity with lobes saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving directivity to file: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    title_str = f"{field_region}\n(z={z_offset_actual:.6f} mm)"

    if args.dB:
        directivity_dB = 20 * np.log10(np.clip(directivity, 1e-12, None))
        ax.plot(theta, directivity_dB, label="Directivity", color="black")
        ax.plot(theta[main_idx], directivity_dB[main_idx], "ro", label="Main Lobe")
        ax.plot(theta[side_lobe_idxs], directivity_dB[side_lobe_idxs], "bo", label="Side Lobes")
        if snell_theta is not None:
            ax.plot(snell_theta, snell_dB, "go", label="Snell Lobe")
            ax.annotate(f"{snell_theta:.1f}°\n{snell_dB:.1f} dB", (snell_theta, snell_dB),
                        textcoords="offset points", xytext=(5, 10), fontsize=20, color="green")
        apply_plot_style(ax, title=title_str, xlabel="Angle (degrees)", ylabel="Amplitude (dB)")
    else:
        ax.plot(theta, directivity, label="Directivity", color="black")
        ax.plot(theta[main_idx], directivity[main_idx], "ro", label="Main Lobe")
        ax.plot(theta[side_lobe_idxs], directivity[side_lobe_idxs], "bo", label="Side Lobes")
        if snell_theta is not None:
            ax.plot(snell_theta, snell_amp, "go", label="Snell Lobe")
            ax.annotate(f"{snell_theta:.1f}°\n{snell_amp:.2f}", (snell_theta, snell_amp),
                        textcoords="offset points", xytext=(5, 10), fontsize=20, color="green")
        apply_plot_style(ax, title=title_str, xlabel="Angle (degrees)", ylabel="Normalized Amplitude")

    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if args.plotfile:
        plt.savefig(args.plotfile, dpi=300, bbox_inches='tight')
        print(f"Directivity plot saved to {args.plotfile}")

    plt.show()

if __name__ == "__main__":
    main()

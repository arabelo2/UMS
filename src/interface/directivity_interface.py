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


# -----------------------------------------------------------------------------#
# Helper styling
# -----------------------------------------------------------------------------#
def apply_plot_style(ax=None, title=None, xlabel=None, ylabel=None):
    if ax is None:
        ax = plt.gca()
    if title:
        ax.set_title(title, fontsize=24)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel:
        ax.set_ylabel(ylabelc)
    ax.tick_params(axis='both', which='major', labelsize=20)


# -----------------------------------------------------------------------------#
# Small utility helpers
# -----------------------------------------------------------------------------#
def find_closest_value(array, target):
    array = np.asarray(array)
    idx = np.searchsorted(array, target, side="left")
    if idx > 0 and (idx == len(array) or abs(target - array[idx - 1]) <= abs(target - array[idx])):
        return array[idx - 1]
    return array[idx]


def compute_directivity(v_slice, x, z_offset, global_max):
    """
    Return angle array (deg) and slice amplitudes
    normalized w.r.t. the GLOBAL peak amplitude of the whole field.
    """
    theta = np.arctan2(x, z_offset) * (180.0 / np.pi)
    directivity = np.abs(v_slice) / global_max
    return theta, directivity


def secondary_analysis(theta, directivity, snell_angle, tolerance=2.0):
    """
    Detect and quantify any weaker lobes near the Snell-predicted angle.
    """
    idx = np.where((theta >= snell_angle - tolerance) & (theta <= snell_angle + tolerance))[0]
    if idx.size == 0:
        return None, None, None
    peak_idx = idx[np.argmax(directivity[idx])]
    peak_amp = directivity[peak_idx]
    peak_dB = 20 * np.log10(np.clip(peak_amp, 1e-12, None))
    return theta[peak_idx], peak_amp, peak_dB


# -----------------------------------------------------------------------------#
# CLI main
# -----------------------------------------------------------------------------#
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required arguments -------------------------------------------------- #
    parser.add_argument("--infile",     required=True,  type=str, help="Path to velocity-field matrix file.")
    parser.add_argument("--x",          required=True,  type=str, help="Range or scalar for x (mm).")
    parser.add_argument("--y",          required=True,  type=str, help="Range or scalar for y (mm).")
    parser.add_argument("--z",          required=True,  type=str, help="Range or scalar for z (mm).")
    parser.add_argument("--z_offset",   required=True,  type=float, help="z slice used for angle calc (mm).")
    parser.add_argument("--freq",       required=True,  type=float, help="Operating frequency (MHz).")
    parser.add_argument("--mat",        required=True,  type=str,
                        help='Material params "d1,cp1,d2,cp2,cs2,wave" (e.g. "1,1480,7.9,5900,3200,p").')

    # --- Optional arguments -------------------------------------------------- #
    parser.add_argument("--snell_angle", type=float, default=None,
                        help="Snell-predicted angle to highlight (deg).")
    parser.add_argument("--outfile",  type=str, default="directivity_output.txt",
                        help="File to save angle / amplitude / lobe-label.")
    parser.add_argument("--plotfile", type=str, default=None,
                        help="Optional filename to save PNG plot.")
    parser.add_argument("--dB", action="store_true", help="Plot in dB instead of linear amplitude.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------#
    # Parse coordinate grids
    x = parse_array(args.x)
    y = parse_array(args.y)
    z = parse_array(args.z)

    # Resolve actual slice depth
    z_offset_actual = find_closest_value(z, args.z_offset)
    print(f"Requested z_offset: {args.z_offset} mm → Using closest z = {z_offset_actual:.6f} mm")

    # Fraunhofer distance (for user info only)
    cp1 = float(args.mat.split(",")[1])        # speed of sound in medium 1 (m/s) but value given in m/s
    wavelength = cp1 / (args.freq * 1e6)       # MHz → Hz
    aperture_D = max(x) - min(x)               # mm
    z_far = (2 * aperture_D**2) / wavelength
    print(f"Fraunhofer distance z_far = {z_far:.3f} mm")

    field_region = "Near-Field Directivity" if z_offset_actual < z_far else "Far-Field Directivity"

    # -------------------------------------------------------------------------#
    # Read velocity field & reshape
    try:
        raw = np.loadtxt(args.infile)
        v = raw.reshape((len(z), len(y), len(x)))
    except Exception as exc:
        sys.exit(f"Error loading / reshaping velocity data: {exc}")

    # Global maximum amplitude for proper normalization
    global_max = np.max(np.abs(v))

    # Take requested z-slice
    idx_z = np.argmin(np.abs(z - z_offset_actual))
    v_slice = v[idx_z, :, :]

    # Reduce to 1-D (x) if y is scalar, else average over y
    v_slice = v_slice.flatten() if len(y) == 1 else np.mean(v_slice, axis=0)

    # -------------------------------------------------------------------------#
    # Compute directivity with GLOBAL normalization
    theta, directivity = compute_directivity(v_slice, x, z_offset_actual, global_max)

    # Main & side lobe detection
    peaks, _ = find_peaks(directivity, height=0.01)  # ignore tiny ripples
    main_idx = np.argmax(directivity)
    side_lobe_idxs = [p for p in peaks if p != main_idx]

    # Snell analysis (optional)
    if args.snell_angle is not None:
        snell_theta, snell_amp, snell_dB = secondary_analysis(theta, directivity, args.snell_angle)
        if snell_theta is not None:
            print(f"Weaker lobe detected near Snell angle ({args.snell_angle}°):")
            print(f"  Angle: {snell_theta:.2f}°, Amplitude: {snell_amp:.3f}, Relative dB: {snell_dB:.2f} dB")
        else:
            print(f"No significant lobe detected near Snell angle ({args.snell_angle}°)")
    else:
        snell_theta = snell_amp = snell_dB = None

    # -------------------------------------------------------------------------#
    # Save text output
    try:
        with open(args.outfile, "w") as f:
            for i, (ang, amp) in enumerate(zip(theta, directivity)):
                label = "main" if i == main_idx else ("side" if i in side_lobe_idxs else "none")
                f.write(f"{ang:.6f}\t{amp:.16f}\t{label}\n")
        print(f"Directivity with lobe labels saved to {args.outfile}")
    except Exception as exc:
        print(f"Error saving directivity data: {exc}")

    # -------------------------------------------------------------------------#
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    title = f"{field_region}\n(z = {z_offset_actual:.6f} mm)"

    if args.dB:
        yvals = 20 * np.log10(np.clip(directivity, 1e-12, None))
        ax.set_ylabel("Amplitude (dB)")
    else:
        yvals = directivity
        ax.set_ylabel("Normalized Amplitude", fontsize=20)

    ax.plot(theta, yvals, "k-", label="Directivity")
    ax.plot(theta[main_idx], yvals[main_idx], "ro", label="Main Lobe")
    ax.plot(theta[side_lobe_idxs], yvals[side_lobe_idxs], "bo", label="Side Lobes")
    if snell_theta is not None:
        marker_val = snell_dB if args.dB else snell_amp
        ax.plot(snell_theta, marker_val, "go", label="Snell Lobe")
        txt = f"{snell_theta:.1f}°\n{snell_dB:.1f} dB" if args.dB else f"{snell_theta:.1f}°\n{snell_amp:.2f}"
        ax.annotate(txt, (snell_theta, marker_val), textcoords="offset points",
                    xytext=(5, 10), fontsize=20, color="green")

    apply_plot_style(ax, title=title, xlabel="Angle (degrees)")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=20)
    plt.tight_layout()

    if args.plotfile:
        plt.savefig(args.plotfile, dpi=300, bbox_inches="tight")
        print(f"Directivity plot saved to {args.plotfile}")

    plt.show()


if __name__ == "__main__":
    main()

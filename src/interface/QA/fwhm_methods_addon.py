# This module implements FWHM estimation methods F2 to F7 based on the paper:
# "Methods for estimating full width at half maximum" by Rainio et al. (2025).
# The input to all methods is a 1D envelope signal (assumed to be already normalized)
# and the corresponding z-axis coordinates.

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import csv
import matplotlib.pyplot as plt

LN2_8 = 8 * np.log(2)

def estimate_fwhm_F2(z_vals, envelope):
    peak_idx = np.argmax(envelope)
    yj = envelope[peak_idx]
    dz = z_vals[1] - z_vals[0]
    N = np.sum(envelope)
    sigma = (1 / np.sqrt(2 * np.pi)) * (N / yj) * dz
    return sigma * np.sqrt(LN2_8)

def estimate_fwhm_F3(z_vals, envelope):
    N = np.sum(envelope)
    T1 = np.sum(z_vals * envelope)
    T2 = np.sum((z_vals ** 2) * envelope)
    mean = T1 / N
    var = T2 / N - mean ** 2
    sigma = np.sqrt(var)
    return sigma * np.sqrt(LN2_8)

def estimate_fwhm_F4(z_vals, envelope):
    mask = envelope > 0.01
    x = z_vals[mask]
    y = np.log(envelope[mask])
    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    if a >= 0:
        return np.nan
    sigma = np.sqrt(-1 / (2 * a))
    return sigma * np.sqrt(LN2_8)

def estimate_fwhm_F5(z_vals, envelope):
    mask = envelope > 0.01
    x = z_vals[mask]
    y = envelope[mask]
    dy = np.diff(np.log(y))
    dx = np.diff(x)
    x_mid = (x[:-1] + x[1:]) / 2
    A, _, _, _, _ = linregress(x_mid, dy / dx)
    sigma = np.sqrt(1 / abs(A))
    return sigma * np.sqrt(LN2_8)

def estimate_fwhm_F6(z_vals, envelope):
    peak_idx = np.argmax(envelope)
    if peak_idx <= 0 or peak_idx >= len(envelope) - 1:
        return np.nan
    x3 = z_vals[peak_idx - 1: peak_idx + 2]
    y3 = envelope[peak_idx - 1: peak_idx + 2]
    coeffs = np.polyfit(x3, y3, 2)
    a, b, c = coeffs
    h = -b**2 / (4 * a) + c if a != 0 else y3[1]
    half = h / 2
    left = np.where(envelope[:peak_idx] < half)[0]
    right = np.where(envelope[peak_idx + 1:] < half)[0] + peak_idx + 1
    if len(left) == 0 or len(right) == 0:
        return np.nan
    l_idx = left[-1]
    r_idx = right[0]
    def interp(idx1, idx2):
        x = z_vals[idx1:idx2 + 1]
        y = envelope[idx1:idx2 + 1]
        coeffs = np.polyfit(x, y - half, 2)
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        return real_roots[np.argsort(np.abs(real_roots - z_vals[idx1]))][0]
    cl = interp(l_idx, l_idx + 1)
    cr = interp(r_idx - 1, r_idx)
    return max(0, cr - cl)

def estimate_fwhm_F7(z_vals, envelope):
    dz = z_vals[1] - z_vals[0]
    N = np.sum(envelope) * dz
    T1 = np.sum(z_vals * envelope) * dz
    T2 = np.sum((z_vals ** 2) * envelope) * dz
    mean = T1 / N
    var = T2 / N - mean ** 2
    init = [mean, np.sqrt(var)]
    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    def cost(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        model = gauss(z_vals, mu, sigma)
        return np.sum((model - envelope)**2)
    bounds = [(z_vals[0], z_vals[-1]), (1e-2, 15.0)]
    result = minimize(cost, init, method='L-BFGS-B', bounds=bounds)
    if not result.success:
        return np.nan
    _, sigma = result.x
    return sigma * np.sqrt(LN2_8) if sigma > 0 else np.nan

def estimate_all_fwhm_methods(z_vals, envelope, fwhm_f1=None, save_csv=False, csv_path="fwhm_comparison.csv", show_plot=False, theoretical_fwhm=29.5):
    peak_idx = np.argmax(envelope)
    peak_z = z_vals[peak_idx]

    mask_broad = (z_vals >= peak_z - 20) & (z_vals <= peak_z + 20)
    mask_narrow = (z_vals >= peak_z - 10) & (z_vals <= peak_z + 10)

    z_broad = z_vals[mask_broad]
    env_broad = envelope[mask_broad]

    z_narrow = z_vals[mask_narrow]
    env_narrow = envelope[mask_narrow]

    results = {
        'F1': fwhm_f1,
        'F2': estimate_fwhm_F2(z_broad, env_broad),
        'F3': estimate_fwhm_F3(z_broad, env_broad),
        'F4': estimate_fwhm_F4(z_narrow, env_narrow),
        'F5': estimate_fwhm_F5(z_narrow, env_narrow),
        'F6': estimate_fwhm_F6(z_broad, env_broad),
        'F7': estimate_fwhm_F7(z_broad, env_broad),
    }

    if save_csv:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'FWHM (mm)', 'Error (%)', 'Status'])
            for k, v in results.items():
                if v is not None and not np.isnan(v):
                    error = abs(v - theoretical_fwhm) / theoretical_fwhm * 100
                    status = '✅ Good' if error < 20 else ('⚠️ Approx' if error < 50 else '❌ Poor')
                    writer.writerow([k, f"{v:.4f}", f"{error:.1f}", status])
                else:
                    writer.writerow([k, 'NaN', 'NaN', '❌ Failed'])

    if show_plot:
        methods = list(results.keys())
        values = [results[m] for m in methods]
        colors = []
        for v in values:
            if v is None or np.isnan(v):
                colors.append('gray')
            else:
                error = abs(v - theoretical_fwhm) / theoretical_fwhm * 100
                if error < 20:
                    colors.append('green')
                elif error < 50:
                    colors.append('orange')
                else:
                    colors.append('red')

        plt.figure(figsize=(10, 5))
        plt.bar(methods, values, color=colors)
        plt.axhline(theoretical_fwhm, color='blue', linestyle='--', label='Theoretical')
        plt.ylabel("FWHM (mm)")
        plt.title("Comparison of FWHM Estimation Methods (F1–F7)")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results

# This module implements FWHM estimation methods F1 to F7 based on the paper:
# "Methods for estimating full width at half maximum" by Rainio et al. (2025).
# The input to all methods is a 1D envelope signal (assumed to be already normalized)
# and the corresponding z-axis coordinates.

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
import csv
import matplotlib.pyplot as plt

LN2_8 = 8 * np.log(2)

# --- Helper for F1 & F6 Linear Interpolation ---
def linear_interp_roots(z_vals, envelope, left_idx, right_idx, target_val):
    """
    Applies Eq (1) linear interpolation from the paper[cite: 51].
    """
    # Left side interpolation
    yl, yl_next = envelope[left_idx], envelope[left_idx+1]
    xl, xl_next = z_vals[left_idx], z_vals[left_idx+1]
    
    # cl = xl + (target - yl)/(yl+1 - yl) * (xl+1 - xl)
    denom_l = yl_next - yl
    if denom_l == 0: cl = xl    
    else: cl = xl + ((target_val - yl) / denom_l) * (xl_next - xl)

    # Right side interpolation
    yr, yr_prev = envelope[right_idx], envelope[right_idx-1]
    xr, xr_prev = z_vals[right_idx], z_vals[right_idx-1]

    # cr = xr - (target - yr)/(yr-1 - yr) * (xr - xr-1)
    denom_r = yr_prev - yr
    if denom_r == 0: cr = xr    
    else: cr = xr + ((target_val - yr) / denom_r) * (xr_prev - xr)

    return cl, cr

def get_middle_most_peak_index(envelope):
    max_val = np.max(envelope)
    max_indices = np.where(envelope == max_val)[0]
    return max_indices[len(max_indices) // 2]

# --- Method F1 (Preserved) ---
def estimate_fwhm_F1(z_vals, envelope):
    j = get_middle_most_peak_index(envelope)
    yj = envelope[j]
    y_half = yj / 2.0

    # Safety: Check if the signal actually crosses the half-max line on both sides
    left_candidates = np.where(envelope[:j] < y_half)[0]
    right_candidates = np.where(envelope[j+1:] < y_half)[0]
    
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        # This is where the 'nan' comes from
        return np.nan

    l = left_candidates[-1]
    r = right_candidates[0] + j + 1

    cl, cr = linear_interp_roots(z_vals, envelope, l, r, y_half)
    return max(0, cr - cl)

# --- Method F2 (Correct) ---
def estimate_fwhm_F2(z_vals, envelope):
    # [cite: 60] Formula based on max height and total area
    peak_idx = np.argmax(envelope)
    yj = envelope[peak_idx]
    if yj == 0: return np.nan
    dz = z_vals[1] - z_vals[0]
    N = np.sum(envelope)
    # sigma = (1 / sqrt(2pi)) * (N / yj) * dz
    sigma = (1 / np.sqrt(2 * np.pi)) * (N / yj) * dz
    return sigma * np.sqrt(LN2_8)

# --- Method F3 (Correct) ---
def estimate_fwhm_F3(z_vals, envelope):
    #  Method of moments
    N = np.sum(envelope)
    if N == 0: return np.nan
    T1 = np.sum(z_vals * envelope)
    T2 = np.sum((z_vals ** 2) * envelope)
    mean = T1 / N
    var = T2 / N - mean ** 2
    if var < 0: return np.nan
    sigma = np.sqrt(var)
    return sigma * np.sqrt(LN2_8)

# --- Method F4 (Correct) ---
def estimate_fwhm_F4(z_vals, envelope):
    #  Fit parabola to log counts
    # Using 0.01 threshold as proxy for "y_i > 3" [cite: 68]
    mask = envelope > 0.01
    if np.sum(mask) < 3: return np.nan
    x = z_vals[mask]
    y = np.log(envelope[mask])
    try:
        coeffs = np.polyfit(x, y, 2)
        a = coeffs[0]
        if a >= 0: return np.nan
        sigma = np.sqrt(-1 / (2 * a))
        return sigma * np.sqrt(LN2_8)
    except:
        return np.nan

# --- Method F5 (Fixed) ---
def estimate_fwhm_F5(z_vals, envelope):
    # [cite: 69-71] Linear regression of derivative of log ratio
    # Requires 2-step difference: ln(y_{i+1}/y_{i-1}) / (x_{i+1}-x_{i-1})
    
    # Filter data first
    indices = np.where(envelope > 0.01)[0]
    if len(indices) < 3: return np.nan

    # Create lists for regression
    X_reg = []
    Y_reg = []

    # We need i-1 and i+1 to be valid indices
    for i in indices:
        if i - 1 < 0 or i + 1 >= len(envelope):
            continue
        
        # [cite: 70] Check if neighbors are also valid (y > 3 condition applies generally)
        if envelope[i-1] <= 0 or envelope[i+1] <= 0:
            continue
            
        x_diff = z_vals[i+1] - z_vals[i-1]
        if x_diff == 0: continue
        
        # Y term: ln(y_{i+1}/y_{i-1}) / (x_{i+1}-x_{i-1})
        val = np.log(envelope[i+1] / envelope[i-1]) / x_diff
        
        X_reg.append(z_vals[i])
        Y_reg.append(val)

    if len(X_reg) < 2: return np.nan

    try:
        # Fit A*x + B
        A, _, _, _, _ = linregress(X_reg, Y_reg)
        # [cite: 71] sigma = sqrt(1 / |A|)
        sigma = np.sqrt(1 / abs(A))
        return sigma * np.sqrt(LN2_8)
    except:
        return np.nan

# --- Method F6 (Fixed) ---
def estimate_fwhm_F6(z_vals, envelope):
    #  NEMA standard method
    peak_idx = get_middle_most_peak_index(envelope)
    
    # 1. Fit parabola to 3 points around max [cite: 73]
    if peak_idx <= 0 or peak_idx >= len(envelope) - 1:
        return np.nan
    
    x3 = z_vals[peak_idx - 1: peak_idx + 2]
    y3 = envelope[peak_idx - 1: peak_idx + 2]
    
    try:
        coeffs = np.polyfit(x3, y3, 2)
        a, b, c = coeffs
        
        # 2. Find peak height h [cite: 74]
        if a == 0: h = y3[1]
        else: h = -b**2 / (4 * a) + c
        
        # 3. Find lh and rh (closest indexes < h/2) [cite: 75]
        half_h = h / 2.0
        
        left_candidates = np.where(envelope[:peak_idx] < half_h)[0]
        if len(left_candidates) == 0: return np.nan
        l_h = left_candidates[-1]
        
        right_candidates = np.where(envelope[peak_idx+1:] < half_h)[0]
        if len(right_candidates) == 0: return np.nan
        r_h = right_candidates[0] + peak_idx + 1
        
        # 4. Use linear interpolation (Eq 1) with h instead of yj [cite: 76]
        cl, cr = linear_interp_roots(z_vals, envelope, l_h, r_h, half_h)
        return max(0, cr - cl)
        
    except:
        return np.nan

# --- Method F7 (Fixed) ---
def estimate_fwhm_F7(z_vals, envelope):
    #  Gaussian fit optimization with scaling and offset
    
    dz = z_vals[1] - z_vals[0]
    
    # 1. Scale vector y so area is approx 1 [cite: 94]
    area = np.trapz(envelope, z_vals)
    if area == 0: return np.nan
    y_prime = envelope / area
    
    # Calculate moments on y' for initialization [cite: 96]
    N_prime = np.sum(y_prime) # Should be approx 1/dz if trapz~1, but using sum as per F3
    T1_prime = np.sum(z_vals * y_prime)
    T2_prime = np.sum((z_vals ** 2) * y_prime)
    
    mean_init = T1_prime / N_prime
    var_init = T2_prime / N_prime - mean_init**2
    sigma_init = np.sqrt(max(1e-6, var_init))
    
    # Initial parameters: A = [A1, A2, A3] -> [offset, mean, sigma]
    # [cite: 96] Initialize A1=0
    init = [0.0, mean_init, sigma_init]
    
    # [cite: 95] Cost function with offset A1
    def cost(params):
        A1, mu, sigma = params
        if sigma <= 0: return np.inf
        
        # Gaussian function f(x | A2, A3)
        f_x = np.exp(-0.5 * ((z_vals - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        # Squared error sum( (f(x) + A1 - y')^2 )
        return np.sum((f_x + A1 - y_prime)**2)

    bounds = [(-np.max(y_prime), np.max(y_prime)), (z_vals[0], z_vals[-1]), (1e-4, (z_vals[-1]-z_vals[0]))]
    
    try:
        # [cite: 97] Optimization
        result = minimize(cost, init, method='L-BFGS-B', bounds=bounds)
        if not result.success:
            return np.nan
            
        _, _, sigma_opt = result.x
        
        # [cite: 98] FWHM = sigma7 * sqrt(8 ln 2)
        return sigma_opt * np.sqrt(LN2_8)
    except:
        return np.nan

def estimate_all_fwhm_methods(z_vals, envelope, theoretical_fwhm=None, fwhm_f1=None, save_csv=False, csv_path="fwhm_comparison.csv", show_plot=False):
    """
    Runs all estimation methods.
    """
    peak_idx = np.argmax(envelope)
    peak_z = z_vals[peak_idx]

    mask_broad = (z_vals >= peak_z - 50) & (z_vals <= peak_z + 50)
    mask_narrow = (z_vals >= peak_z - 10) & (z_vals <= peak_z + 10)

    z_broad = z_vals[mask_broad]
    env_broad = envelope[mask_broad]

    z_narrow = z_vals[mask_narrow]
    env_narrow = envelope[mask_narrow]
    
    # Use internal F1 if not provided, otherwise use passed value
    if theoretical_fwhm is None:
        theoretical_fwhm = estimate_fwhm_F1(z_broad, env_broad)
    else:
        theoretical_fwhm = theoretical_fwhm
    
    # Use internal F1 if not provided, otherwise use passed value
    if fwhm_f1 is None:
        calculated_f1 = estimate_fwhm_F1(z_broad, env_broad)
    else:
        calculated_f1 = fwhm_f1

    results = {
        'F1': calculated_f1,
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
                if v is not None and not np.isnan(v) and theoretical_fwhm is not None and theoretical_fwhm != 0:
                    error = abs(v - theoretical_fwhm) / theoretical_fwhm * 100
                    status = '✅ Good' if error < 10 else ('⚠️ Approx' if error < 50 else '❌ Poor')
                    writer.writerow([k, f"{v:.4f}", f"{error:.1f}", status])
                else:
                    writer.writerow([k, f"{v if v is not None else 'NaN'}", 'NaN', '❌ Failed'])

    if show_plot:
        methods = list(results.keys())
        values = [results[m] if results[m] is not None else np.nan for m in methods]
        colors = []
        for v in values:
            if v is None or np.isnan(v):
                colors.append('gray')
            elif theoretical_fwhm is not None and theoretical_fwhm != 0:
                error = abs(v - theoretical_fwhm) / theoretical_fwhm * 100
                if error < 10:
                    colors.append('green')
                elif error < 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            else:
                colors.append('blue')

        plt.figure(figsize=(10, 5))
        plt.bar(methods, values, color=colors)
        if theoretical_fwhm:
            plt.axhline(theoretical_fwhm, color='blue', linestyle='--', label='Theoretical')
        plt.ylabel("FWHM (mm)")
        plt.title("Comparison of FWHM Estimation Methods (F1–F7)")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results
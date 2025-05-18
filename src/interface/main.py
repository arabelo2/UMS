#!/usr/bin/env python3

"""
FMC/TFM simulation for the ASTM E2491 Type B assessment block - version with
**time-gate applied in the FMC domain** (8µs default).
Only the following changes were made relative to the previous canvas version:
1. `generate_fmc_data()` now takes an optional `gate_us` parameter and, if
   provided, zeros the first *gate_us x fs* samples of the FMC cube.
2. Both example runs call `generate_fmc_data(..., gate_us=8.0)`.
3. O *gate* manual na imagem foi removido.
4. Normalização corrigida em `env2 / env2.max()`.
No other logic was modified.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gausspulse, hilbert

# ---------------------------------------------------------------------------
# external services (path hack kept as‑is)
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application.ls_2Dint_service import run_ls_2Dint_service
from application.discrete_windows_service import run_discrete_windows_service

# ===========================================================
# 1. DEFECT EMULATION (Modular Function)
# ===========================================================

INCH = 0.0254  # m

def simulate_defect(defect_type, **kwargs):
    """
    Return arrays (x, z) of point‑scatterer coordinates in metres that will be
    treated as defects / reflectors by the rest of the pipeline.
    """
    if defect_type == 'A':
        x = np.linspace(-0.005, 0.005, kwargs.get('n_points', 50))
        z = np.full_like(x, 0.035)

    elif defect_type == 'B':
        z = np.linspace(0.030, 0.040, kwargs.get('n_points', 50))
        x = np.full_like(z, 0.0)

    elif defect_type == 'C':
        angle_rad = np.radians(kwargs.get('angle_deg', 45))
        x = (np.linspace(0, kwargs.get('length', 0.01)*np.cos(angle_rad),
                         kwargs.get('n_points', 50))
             + kwargs.get('x_start', -0.005))
        z = (np.linspace(0, kwargs.get('length', 0.01)*np.sin(angle_rad),
                         kwargs.get('n_points', 50))
             + kwargs.get('z_start', 0.030))

    elif defect_type == 'D':
        theta = np.linspace(0, 2*np.pi, kwargs.get('n_points', 50))
        centre = kwargs.get('center', (0.0, 0.035))
        radius = kwargs.get('radius', 0.001)
        x = centre[0] + radius * np.cos(theta)
        z = centre[1] + radius * np.sin(theta)

    elif defect_type == 'G':
        theta = np.radians(
            np.linspace(*kwargs.get('theta_range', (225, 315)),
                        kwargs.get('n_points', 15)))
        centre = kwargs.get('center', (0.005, 0.035))
        radius = kwargs.get('radius', 0.010)
        x = centre[0] + radius * np.cos(theta)
        z = centre[1] - radius * np.sin(theta)

    # ------------------------------------------------------------------
    # NEW: full ASTM E2491 Type B reflector set
    # ------------------------------------------------------------------
    elif defect_type == 'ASTM':
        # ---- parameters taken from the block drawing (inches → metres) ----
        arc_center = (1.0 * INCH, 2.0 * INCH)          # X, Z of arc origin
        arc_radii  = [1.0 * INCH, 2.0 * INCH]          # two arcs
        arc_angles = np.deg2rad(np.arange(-45, 46, 5)) # −45° … +45° in 5° steps

        # 1) two radial hole arcs (18 + 18 points)
        radial_x, radial_z = [], []
        for r in arc_radii:
            radial_x.append(arc_center[0] + r * np.cos(arc_angles))
            radial_z.append(arc_center[1] + r * np.sin(arc_angles))
        radial_x = np.concatenate(radial_x)
        radial_z = np.concatenate(radial_z)

        # 2) vertical column (16 points, 0.120‑in pitch)
        col_x = np.full(16, 4.0 * INCH)
        col_z = 0.5 * INCH + 0.120 * INCH * np.arange(16)

        # 3) 30° angled row (12 points, 0.200‑in pitch)
        row_start = np.array([1.0 * INCH, 0.5 * INCH])
        row_dir   = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
        row_pts   = row_start + (0.200 * INCH) * np.arange(12).reshape(-1, 1) * row_dir
        row_x, row_z = row_pts[:, 0], row_pts[:, 1]

        # pack everything together
        x = np.concatenate([radial_x, col_x, row_x])
        z = np.concatenate([radial_z, col_z, row_z])

        # allow an optional random shuffle to avoid imaging artefacts that
        # arise when scatterers are in a regular grid
        if kwargs.get('shuffle', False):
            idx = np.random.permutation(len(x))
            x, z = x[idx], z[idx]

    # ------------------------------------------------------------------
    elif defect_type == 'NONE':
        x = np.array([]);  z = np.array([])

    elif defect_type == 'SINGLE':
        x = np.array([0.0]);  z = np.array([0.035])

    else:
        raise ValueError(f"Unknown defect type: {defect_type}")

    return x, z

# ===========================================================
# 2. ULTRASOUND FIELD SETUP (Configurations)
# ===========================================================
def configure_ultrasound_system():
    """Define transducer array and pulse parameters."""
    c1, c2 = 1480, 5900  # Wave speeds (m/s) for water and steel
    z_interface = 0.010   # Interface depth (m)

    # Transducer array
    pitch = 0.0007        # Element spacing (m)
    num_elements = 64     # Number of elements
    element_pos = np.linspace(-(num_elements-1)/2*pitch, (num_elements-1)/2*pitch, num_elements)
    
    # Ângulo da cunha em graus
    angt = 0

    # Excitation pulse
    fc = 5e6              # Center frequency (Hz)
    bw = 0.8              # Bandwidth
    duration = 5 / (bw * fc)
    t_pulse = np.arange(-duration/2, duration/2, 1/100e6)
    pulse = gausspulse(t_pulse, fc=fc, bw=bw, bwr=-6)
    pulse /= np.max(np.abs(pulse))

    return {
        'c1': c1, 'c2': c2, 'z_interface': z_interface,
        'pitch': pitch, 'num_elements': num_elements, 'element_pos': element_pos,
        'pulse': pulse, 'fs': 100e6, 'num_samples': 8192, 'fc': fc,
        'angt': angt
    }

# ===========================================================
# 3. FMC DATA GENERATION (Core Logic)
# ===========================================================

def generate_fmc_data(defect_x, defect_z, config, *,
                      use_ls2dint=True,
                      noise_snr_db=None,
                      gate_us=None):
    """Return full FMC cube; optional *gate_us* removes early-time data."""
    
    fmc = np.zeros((config['num_samples'],
                    config['num_elements'],
                    config['num_elements']))
    
    window = run_discrete_windows_service(config['num_elements'], 'Ham')
    fc, pitch = config['fc'], config['pitch']
    c1, c2    = config['c1'], config['c2']

    for tx in range(config['num_elements']):
        for rx in range(config['num_elements']):
            for xi, zi in zip(defect_x, defect_z):
                amp = np.abs(run_ls_2Dint_service(
                    b=0.5 * pitch * 1e3, f=fc * 1e-6, c=c1,
                    e=config['element_pos'][tx] * 1e3,
                    mat=[1.0, c1, 7.9, c2],
                    angt=config['angt'], Dt0=config['z_interface'] * 1e3,
                    x=xi * 1e3, z=zi * 1e3, Nopt=20)) if use_ls2dint else 1.0
                idx = int(calculate_tof(xi, zi, tx, rx, config) * config['fs'])
                if 0 <= idx < config['num_samples'] - len(config['pulse']):
                    fmc[idx:idx + len(config['pulse']), tx, rx] += (
                        amp * config['pulse'] * window[tx] * window[rx])

    # --- time‑gate on the FMC ------------------------------------------

    if gate_us is not None:
        gate_samples = int(gate_us * 1e-6 * config['fs'])
        fmc[:gate_samples, :, :] = 0

    if noise_snr_db:
        fmc = add_noise(fmc, noise_snr_db)
    return fmc

def calculate_tof(x, z, tx, rx, config):
    """Helper: Compute time-of-flight for a scatterer at (x,z) with refraction."""
    # Transmitter path
    dx_tx = x - config['element_pos'][tx]
    incident_angle_rad = np.radians(config.get('angt', 0.0))  # 30 graus da cunha
    dx_tx_rot = dx_tx * np.cos(incident_angle_rad) - config['z_interface'] * np.sin(incident_angle_rad)
    dz_tx_rot = dx_tx * np.sin(incident_angle_rad) + config['z_interface'] * np.cos(incident_angle_rad)
    theta1_tx = np.arctan2(dx_tx_rot, dz_tx_rot)

    theta2_tx = np.arcsin(np.clip((config['c1'] / config['c2']) * np.sin(theta1_tx), -1, 1))
    l1_tx = config['z_interface'] / np.cos(theta1_tx)
    l2_tx = (z - config['z_interface']) / np.cos(theta2_tx)
    
    # Receiver path (similar logic)
    dx_rx = x - config['element_pos'][rx]
    dx_rx_rot = dx_rx * np.cos(incident_angle_rad) - config['z_interface'] * np.sin(incident_angle_rad)
    dz_rx_rot = dx_rx * np.sin(incident_angle_rad) + config['z_interface'] * np.cos(incident_angle_rad)
    theta1_rx = np.arctan2(dx_rx_rot, dz_rx_rot)  
    theta2_rx = np.arcsin(np.clip((config['c1'] / config['c2']) * np.sin(theta1_rx), -1, 1))
    l1_rx = config['z_interface'] / np.cos(theta1_rx)
    l2_rx = (z - config['z_interface']) / np.cos(theta2_rx)
    
    return (l1_tx + l1_rx)/config['c1'] + (l2_tx + l2_rx)/config['c2']

def add_noise(signal, snr_db):
    """Add Gaussian noise to FMC data."""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# ===========================================================
# 4. TFM RECONSTRUCTION (Imaging)
# ===========================================================
def reconstruct_tfm(fmc_data, config, x_img, z_img):
    """Delay-and-sum beamforming to create TFM image."""
    X, Z = np.meshgrid(x_img, z_img)
    tfm_image = np.zeros_like(X)
    
    for tx in range(config['num_elements']):
        for rx in range(config['num_elements']):
            tof_total = calculate_tof_grid(X, Z, tx, rx, config)
            idx = np.round(tof_total * config['fs']).astype(int)
            valid = (idx >= 0) & (idx < config['num_samples'])
            tfm_image[valid] += fmc_data[idx[valid], tx, rx]
    
    return tfm_image / np.max(np.abs(tfm_image)) if np.max(np.abs(tfm_image)) != 0 else tfm_image

def calculate_tof_grid(X, Z, tx, rx, config):
    """Vectorized ToF calculation for TFM grid."""
    # Transmitter path
    dx_tx = X - config['element_pos'][tx]
    dz_tx = Z
    theta1_tx = np.arctan2(dx_tx, dz_tx)
    sin_theta2_tx = (config['c1'] / config['c2']) * np.sin(theta1_tx)
    theta2_tx = np.arcsin(np.clip(sin_theta2_tx, -1, 1))
    l1_tx = np.minimum(config['z_interface'], dz_tx) / np.cos(theta1_tx)
    l2_tx = np.where(dz_tx > config['z_interface'], (dz_tx - config['z_interface']) / np.cos(theta2_tx), 0)
    tof_tx = np.where(dz_tx <= config['z_interface'], l1_tx/config['c1'], l1_tx/config['c1'] + l2_tx/config['c2'])
    
    # Receiver path (similar)
    dx_rx = X - config['element_pos'][rx]
    dz_rx = Z
    theta1_rx = np.arctan2(dx_rx, dz_rx)
    sin_theta2_rx = (config['c1'] / config['c2']) * np.sin(theta1_rx)
    theta2_rx = np.arcsin(np.clip(sin_theta2_rx, -1, 1))
    l1_rx = np.minimum(config['z_interface'], dz_rx) / np.cos(theta1_rx)
    l2_rx = np.where(dz_rx > config['z_interface'], (dz_rx - config['z_interface']) / np.cos(theta2_rx), 0)
    tof_rx = np.where(dz_rx <= config['z_interface'], l1_rx/config['c1'], l1_rx/config['c1'] + l2_rx/config['c2'])
    
    return tof_tx + tof_rx

# ---------------------------------------------------------------------------
# 5. MAIN EXECUTION EXAMPLE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = configure_ultrasound_system()

    x_img = np.linspace(-0.130, 0.130, 701)
    z_img = np.linspace(0.001, 0.120, 480)

    # ---- first run: full ASTM block with noise & gate ------------------
    defect_x, defect_z = simulate_defect('ASTM', shuffle=False)
    fmc = generate_fmc_data(defect_x, defect_z, cfg,
                            use_ls2dint=True,
                            noise_snr_db=20,
                            gate_us=8.0)
    tfm  = reconstruct_tfm(fmc, cfg, x_img, z_img)
    env  = np.abs(hilbert(tfm, axis=0))

    plt.figure(figsize=(10, 8))
    plt.imshow(20 * np.log10(env / env.max() + 1e-6), cmap='jet',
               extent=[x_img.min()*1e3, x_img.max()*1e3,
                       z_img.max()*1e3, z_img.min()*1e3],
               aspect='equal', vmin=-15, vmax=0)
    plt.scatter(defect_x * 1e3, defect_z * 1e3, c='white', s=10,
                label='Reflectors'); plt.legend()
    plt.colorbar(label='Amplitude (dB)')
    plt.title('TFM – ASTM E2491, FMC gate 8 µs, SNR 20 dB')
    plt.tight_layout(); plt.show()

    # ---- second run: same reflectors, sem ruído ------------------------
    fmc2 = generate_fmc_data(defect_x, defect_z, cfg,
                             use_ls2dint=True,
                             noise_snr_db=None,
                             gate_us=8.0)
    tfm2 = reconstruct_tfm(fmc2, cfg, x_img, z_img)
    env2 = np.abs(hilbert(tfm2, axis=0))

    plt.figure(figsize=(10, 8))
    plt.imshow(20 * np.log10(env2 / env2.max() + 1e-6), cmap='jet',
               extent=[x_img.min()*1e3, x_img.max()*1e3,
                       z_img.max()*1e3, z_img.min()*1e3],
               aspect='equal', vmin=-15, vmax=0)
    plt.scatter(defect_x * 1e3, defect_z * 1e3, c='white', s=10)
    plt.colorbar(label='Amplitude (dB)')
    plt.title('TFM – ASTM E2491, FMC gate 8 µs, sem ruído')
    plt.tight_layout(); plt.show()

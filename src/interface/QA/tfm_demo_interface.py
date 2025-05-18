#!/usr/bin/env python3
# interface/tfm_demo_interface.py
#
# Quick FMC → TFM pipeline that re‑uses your existing code base.
#
# * Builds the array geometry with ElementsCalculator
# * Generates a crude frequency‑domain FMC matrix
# * Applies a frequency‑domain TFM delay‑and‑sum
# * Plots the reconstructed image
#
# ----------------------------------------------------------------------

import os, sys, numpy as np, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 1.  project‑local helpers ----------------------------------------
from domain.elements import ElementsCalculator
from application.delay_laws2D_service import run_delay_laws2D_service
from application.discrete_windows_service import run_discrete_windows_service

# ----------------------------------------------------------------------
# USER‑TUNABLE PARAMETERS
# ----------------------------------------------------------------------
f_MHz   = 5.0            # centre frequency [MHz]
c_ms    = 1480.0         # sound speed [m s‑1]
M       = 32             # number of elements
dl      = 0.5            # element length / λ
gd      = 0.10           # gap / element length
wtype   = 'rect'         # apodisation window
roi_x   = np.linspace(-20, 20, 401)   # mm
roi_z   = np.linspace(  5, 80, 401)   # mm
# ----------------------------------------------------------------------

# --- 2.  array geometry -----------------------------------------------
calc          = ElementsCalculator(f_MHz, c_ms, dl, gd, M)
A, d, g, e_mm = calc.calculate()          # centroids [mm], element length d, gap g
e_m = e_mm * 1e-3                         # mm → m     (needed for k·r)
k = 2 * np.pi * (f_MHz * 1e6) / c_ms      # spatial wavenumber [rad m‑1]
pitch         = d + g                     # element pitch [mm]
wtype = 'Han'
window_amp = run_discrete_windows_service(M, wtype)

# --- 3. synthetic FMC (single‑frequency) ------------------------------
FMC = np.zeros((M, M), dtype=complex)   #  <‑‑‑‑‑ put this back
eps = 1e-6                              # 1 µm in metres, avoids div‑by‑0
for tx in range(M):
    for rx in range(M):
        r = abs(e_m[tx] - e_m[rx]) + eps
        FMC[tx, rx] = (1.0 / r) * np.exp(-1j * k * r)


# --- 4.  reconstruction grid ------------------------------------------
xx, zz = np.meshgrid(roi_x, roi_z)        # mm
xx_m   = xx * 1e-3                        # → m
zz_m   = zz * 1e-3

# --- 5.  frequency‑domain TFM -----------------------------------------
img = np.zeros_like(xx, dtype=complex)

for tx in range(M):
    for rx in range(M):

        # path lengths Tx‑>pixel and pixel‑>Rx [m]
        r_tx = np.sqrt((xx_m - e_m[tx])**2 + zz_m**2)
        r_rx = np.sqrt((xx_m - e_m[rx])**2 + zz_m**2)

        # two‑way phase term
        phase = np.exp(1j * k * (r_tx + r_rx))

        # crude amplitude factor (1/r fall‑off); swap in run_ls_2Dv_service
        amp = 1.0 / (r_tx + r_rx)

        img += window_amp[tx] * window_amp[rx] * FMC[tx, rx] * amp * phase

# --- 6. display -------------------------------------------------------
img_db = 20*np.log10(np.abs(img) / np.max(np.abs(img)))

plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(
    roi_x, roi_z, img_db,
    shading='auto', cmap='jet', vmin=-40, vmax=0      # <- no  aspect=
)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')        # <- do it here instead
plt.xlabel('x  [mm]')
plt.ylabel('z  [mm]')
plt.title(f'Synthetic TFM image  (f = {f_MHz} MHz,  M = {M})')
plt.colorbar(pcm, label='Amplitude [dB]')
plt.tight_layout()
plt.show()

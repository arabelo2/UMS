#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Configuration
results_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../results/tfm_synthetic_flaw'))
tx_idx = rx_idx = 16  # Center element

# Load A-scan
df = pd.read_csv(os.path.join(results_dir, f"fmc_tx{tx_idx}_rx{rx_idx}.csv"))
time_us, rf = df['Time (us)'].values, df['Signal'].values
env = np.abs(hilbert(rf))

# Plot
plt.figure(figsize=(10,5))
plt.plot(time_us, rf, label=f"A-scan RF (el {tx_idx}→{rx_idx})")
plt.plot(time_us, env, label="Hilbert Envelope", linewidth=2)
plt.axvline(2*(30/1000)/5900*1e6, color='r', linestyle='--', label='Expected Flaw TOF')
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Single A-scan RF vs. Envelope (Tx={tx_idx}, Rx={rx_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'single_ascan_plot.png'))
plt.close()
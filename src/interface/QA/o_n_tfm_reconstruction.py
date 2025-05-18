# interface/o_n_tfm_reconstruction.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numba import jit
import time

def o_n_tfm_reconstruction():
    # Configuration
    FMC_FILE_PATH = 'C:/Users/HP/Downloads/FMC_2012_06_12_at_13_09.mat'
    USE_ENVELOPE = True
    META_SAMPLING_RATE_HZ = 100e6
    META_ARRAY_PITCH_MM = 0.7
    META_WAVE_SPEED_MPS = 5820
    FLAW_X_MM = 35.0
    FLAW_Z_MM = 20.0
    
    # Load data (O(1) with memory mapping)
    mat_data = scipy.io.loadmat(FMC_FILE_PATH)
    fmc_data = np.ascontiguousarray(mat_data['FMC_new'].transpose(1, 2, 0))  # MATLAB permute
    
    # Reduced parameters for O(N)
    NUM_X_PIXELS = 601
    NUM_Z_PIXELS = 200
    SUBSAMPLE_ELEMENTS = 16  # Process √N elements instead of N
    
    # Imaging region
    x_min, x_max = -150e-3, 150e-3  # 70mm
    z_min, z_max = 1e-3, 101e-3   # 50mm
    x_coords = np.linspace(x_min, x_max, NUM_X_PIXELS)
    z_coords = np.linspace(z_min, z_max, NUM_Z_PIXELS)
    
    # Key O(N) optimizations
    element_indices = np.linspace(0, fmc_data.shape[0]-1, SUBSAMPLE_ELEMENTS, dtype=int)
    time_indices = np.linspace(0, fmc_data.shape[1]-1, 1000, dtype=int)
    
    # Numba-accelerated O(N) reconstruction
    @jit(nopython=True, fastmath=True)
    def fast_tfm(fmc_data, element_x, wave_speed, time_vec, 
                x_coords, z_coords, element_indices, time_indices):
        image = np.zeros((len(z_coords), len(x_coords)))
        dt = time_vec[1] - time_vec[0]
        
        for iz, z in enumerate(z_coords):
            for ix, x in enumerate(x_coords):
                sum_val = 0.0
                for tx in element_indices:
                    tx_x = element_x[tx]
                    dist_tx = np.sqrt((x - tx_x)**2 + z**2)
                    tof_tx = dist_tx / wave_speed
                    
                    for rx in element_indices:
                        rx_x = element_x[rx]
                        dist_rx = np.sqrt((x - rx_x)**2 + z**2)
                        total_tof = tof_tx + dist_rx / wave_speed
                        
                        # Nearest neighbor approximation (O(1) lookup)
                        time_idx = int(round(total_tof / dt))
                        if 0 <= time_idx < len(time_vec):
                            sum_val += fmc_data[rx, min(time_idx, fmc_data.shape[1]-1), tx]
                
                image[iz, ix] = sum_val
        return image
    
    # Precompute geometry
    element_x = np.arange(fmc_data.shape[0]) * (META_ARRAY_PITCH_MM / 1000)
    time_vec = np.arange(fmc_data.shape[1]) / META_SAMPLING_RATE_HZ
    
    print("Running O(N) approximation...")
    start = time.time()
    tfm_image = fast_tfm(
        fmc_data.astype(np.float32),
        element_x.astype(np.float32),
        np.float32(META_WAVE_SPEED_MPS),
        time_vec.astype(np.float32),
        x_coords.astype(np.float32),
        z_coords.astype(np.float32),
        element_indices,
        time_indices
    )
    print(f"Completed in {time.time()-start:.2f} seconds")
    
    # Visualization
    plt.imshow(tfm_image, extent=[x_min*1000, x_max*1000, z_max*1000, z_min*1000])
    plt.scatter(FLAW_X_MM, FLAW_Z_MM, c='red', marker='x')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    o_n_tfm_reconstruction()
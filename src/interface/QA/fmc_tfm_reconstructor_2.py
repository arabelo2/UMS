# interface/fmc_tfm_reconstructor_2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def load_and_preprocess_data(file_path, use_envelope=True):
    """Load and preprocess data with memory-efficient operations"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        fmc_data = mat_data['FMC_new']
        
        # MATLAB-style permutation: [Tx×Rx×Nt] -> [Rx×Nt×Tx]
        fmc_data = np.transpose(fmc_data, (1, 2, 0))  # Equivalent to MATLAB's permute(raw, [2 3 1])
        
        # Convert to float32 to save memory (unless we need complex)
        fmc_data = fmc_data.astype(np.float32 if use_envelope else np.complex64)
        
        # Process envelope in chunks if needed
        if use_envelope:
            print("Computing envelope in chunks...")
            chunk_size = 32  # Process 32 receivers at a time
            num_rx = fmc_data.shape[0]
            envelope_data = np.zeros_like(fmc_data)
            
            for start in range(0, num_rx, chunk_size):
                end = min(start + chunk_size, num_rx)
                print(f"Processing receivers {start}-{end-1}...")
                envelope_data[start:end] = np.abs(scipy.signal.hilbert(
                    fmc_data[start:end], axis=1))  # Hilbert along time axis
            return envelope_data, mat_data
            
        return fmc_data, mat_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def precompute_geometry(num_elements, array_pitch, wave_speed):
    """Precompute element positions with reduced precision"""
    element_x = np.arange(num_elements, dtype=np.float32) * array_pitch
    element_z = np.zeros(num_elements, dtype=np.float32)
    return element_x, element_z, wave_speed

def compute_pixel_chunk(args):
    """Process a chunk of pixels to reduce memory overhead"""
    chunk, fmc_data, time_vector, element_x, wave_speed = args
    results = np.zeros(len(chunk), dtype=np.float32)
    
    for i, (iz, ix, xp, zp) in enumerate(chunk):
        # Vectorized distance calculations
        tx_dist = np.sqrt((xp - element_x)**2 + zp**2)
        rx_dist = tx_dist  # Symmetrical for this geometry
        total_time = (tx_dist + rx_dist) / wave_speed
        
        # Find nearest time indices (no interpolation to save memory)
        time_indices = np.searchsorted(time_vector, total_time)
        time_indices = np.clip(time_indices, 0, len(time_vector)-1)
        
        # Sum contributions from all tx-rx pairs
        results[i] = np.sum(fmc_data[np.arange(len(element_x)), time_indices, np.arange(len(element_x))])
    
    return results

def reconstruct_tfm(fmc_data, time_vector, element_x, wave_speed, 
                   x_coords, z_coords, num_workers=2):
    """Memory-efficient reconstruction with chunked processing"""
    num_z, num_x = len(z_coords), len(x_coords)
    tfm_image = np.zeros((num_z, num_x), dtype=np.float32)
    
    # Create pixel coordinates
    pixels = [(iz, ix, x_coords[ix], z_coords[iz]) 
             for iz in range(num_z) for ix in range(num_x)]
    
    # Process in chunks to limit memory usage
    chunk_size = 1000  # Pixels per chunk
    chunks = [pixels[i:i + chunk_size] for i in range(0, len(pixels), chunk_size)]
    
    # Setup parallel processing
    pixel_func = partial(compute_pixel_chunk,
                        fmc_data=fmc_data,
                        time_vector=time_vector,
                        element_x=element_x,
                        wave_speed=wave_speed)
    
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap(pixel_func, chunks)):
            start_idx = i * chunk_size
            end_idx = start_idx + len(result)
            for res_idx, (iz, ix, _, _) in enumerate(chunks[i]):
                tfm_image[iz, ix] = result[res_idx]
            
            # Progress feedback
            if (i + 1) % max(1, len(chunks) // 10) == 0:
                print(f"Processed {min(end_idx, len(pixels))}/{len(pixels)} pixels")
    
    return tfm_image

def main():
    # Configuration - adjusted for your hardware
    FMC_FILE_PATH = 'C:/Users/HP/Downloads/FMC_2012_06_12_at_13_09.mat'
    USE_ENVELOPE = True
    META_SAMPLING_RATE_HZ = 100e6
    META_NUM_SAMPLE_POINTS = 10000
    META_ARRAY_PITCH_MM = 0.7
    META_WAVE_SPEED_MPS = 5820
    META_NUM_ELEMENTS = 128
    FLAW_X_MM = 35.0
    FLAW_Z_MM = 20.0
    
    # Hardware-aware parameters
    NUM_X_PIXELS = 80  # Further reduced for memory safety
    NUM_Z_PIXELS = 60
    NUM_WORKERS = 2  # Conservative for your 8GB RAM
    
    print("Loading and preprocessing data...")
    fmc_data, mat_data = load_and_preprocess_data(FMC_FILE_PATH, USE_ENVELOPE)
    
    # Get or generate time vector (float32 to save memory)
    time_vector = mat_data.get('time_vector', 
                             np.arange(fmc_data.shape[1], dtype=np.float32) / 
                             np.float32(META_SAMPLING_RATE_HZ)).squeeze()
    
    # Get geometry parameters
    array_pitch = np.float32(mat_data.get('array_pitch', META_ARRAY_PITCH_MM / 1000.0))
    wave_speed = np.float32(mat_data.get('wave_speed', META_WAVE_SPEED_MPS))
    num_elements = fmc_data.shape[0]
    
    # Precompute geometry
    element_x, element_z, wave_speed = precompute_geometry(
        num_elements, array_pitch, wave_speed)
    
    # Define imaging region
    x_min = max(-0.005, FLAW_X_MM/1000 - 0.020)
    x_max = min(element_x[-1] + 0.010, FLAW_X_MM/1000 + 0.040)
    z_min = 0.001
    z_max = FLAW_Z_MM/1000 + 0.030
    
    x_coords = np.linspace(x_min, x_max, NUM_X_PIXELS, dtype=np.float32)
    z_coords = np.linspace(z_min, z_max, NUM_Z_PIXELS, dtype=np.float32)
    
    print(f"Reconstructing {NUM_X_PIXELS}x{NUM_Z_PIXELS} image using {NUM_WORKERS} workers...")
    start_time = time.time()
    
    tfm_image = reconstruct_tfm(fmc_data, time_vector, element_x, wave_speed,
                              x_coords, z_coords, NUM_WORKERS)
    
    elapsed = time.time() - start_time
    print(f"Reconstruction completed in {elapsed:.2f} seconds")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(tfm_image, aspect='auto',
              extent=[x_min*1000, x_max*1000, z_max*1000, z_min*1000],
              cmap='gray')
    plt.scatter(FLAW_X_MM, FLAW_Z_MM, c='red', marker='x', s=100)
    plt.colorbar(label='Intensity')
    plt.title(f"Memory-Optimized TFM Reconstruction\n{NUM_X_PIXELS}x{NUM_Z_PIXELS} resolution")
    plt.xlabel("X position (mm)")
    plt.ylabel("Z position (mm)")
    plt.show()

if __name__ == "__main__":
    main()
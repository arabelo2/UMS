# test_mps_service_v2.py
import sys
import os
import numpy as np
import datetime

# --- Python Path Setup (same as before) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
interface_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(interface_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
# --- END ---

from src.application.mps_array_model_int_service import run_mps_array_model_int_service
# Assuming parse_array is in cli_utils and cli_utils is in src.interface
from src.interface.cli_utils import parse_array # Make sure this matches your actual parse_array

print(f"[{datetime.datetime.now()}] Test script V2 started.")

# Minimal parameters matching interface_A defaults where L1=1, L2=1 is forced
lx, ly, gx, gy = 0.45, 10.0, 0.05, 0.0 # From your master_pipeline call
f_mhz = 5.0
f_hz = f_mhz * 1e6
d1, c1 = 1.0, 1480.0
d2, c2, cs2 = 7.9, 5900.0, 3200.0
wave_type = 'p'
L1_serv, L2_serv = 1, 1 # Minimal elements
angt = 0.0 # From your master_pipeline call
Dt0 = 20.0 # From your master_pipeline call
theta20 = 0.0 # From your master_pipeline call
phi = 0.0 # From your master_pipeline call
DF = 30.0 # From your master_pipeline call
ampx_type, ampy_type = 'rect', 'rect'

# Simulate how interface_A would parse single point inputs
# For interface_A, xs="0", zs="20", y="0" would be for a true single point.
# Your master_pipeline test uses "0,0,1" which parse_array/parse_scan_vector makes np.array([0.])
# Let's use the "start,stop,num_points=1" convention
xs_str = "0,0,1"
zs_str = "20,20,1"
y_str = "0,0,1" # If y should also be a vector input for consistency with service handling

try:
    # Using the parse_array from interface_A's context if possible,
    # or your master_pipeline's parse_scan_vector
    # Forcing simple array creation here for directness:
    xs_test = np.array([0.0], dtype=np.float32)
    zs_test = np.array([20.0], dtype=np.float32)
    y_test_for_service = np.array([0.0], dtype=np.float32) # Pass as array, MPSArrayModelInt will handle it

except Exception as e:
    print(f"Error parsing test arrays: {e}")
    sys.exit(1)

print(f"[{datetime.datetime.now()}] Parameters set.")
print(f"  xs_test (to service): {xs_test} (shape {xs_test.shape}, type {xs_test.dtype})")
print(f"  zs_test (to service): {zs_test} (shape {zs_test.shape}, type {zs_test.dtype})")
print(f"  y_test_for_service (to service): {y_test_for_service} (shape {y_test_for_service.shape}, type {y_test_for_service.dtype})")

print(f"[{datetime.datetime.now()}] Calling run_mps_array_model_int_service...")

try:
    result = run_mps_array_model_int_service(
        lx, ly, gx, gy,
        f_hz, d1, c1, # service expects f in Hz from its own example, master_pipeline passes f*1e6
        d2, c2, cs2,
        wave_type, L1_serv, L2_serv,
        angt, Dt0, theta20,
        phi, DF,
        ampx_type,
        ampy_type,
        xs_test, zs_test, y_test_for_service # Pass the prepared arrays
    )
    print(f"[{datetime.datetime.now()}] run_mps_array_model_int_service finished.")
    # ... (result printing as before) ...
    if isinstance(result, dict) and 'p' in result:
        print(f"Result['p'] shape: {result['p'].shape}, dtype: {result['p'].dtype}, value: {result['p']}")
    else:
        print(f"Unexpected result content: {result}")


except Exception as e:
    print(f"[{datetime.datetime.now()}] ERROR during service call or processing:")
    import traceback
    traceback.print_exc()

print(f"[{datetime.datetime.now()}] Test script V2 finished.")
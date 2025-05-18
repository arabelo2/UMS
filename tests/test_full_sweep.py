'''
tests/test_full_sweep.py
'''
import sys, os

# application layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.interface.pipeline_tfm_backprojection_demo import compute_fmc, tfm_backprojection

def test_tfm_fluid_solid_interface():
    # 1) set up a two-layer medium: c1=1500 m/s (fluid), c2=3000 m/s (solid), interface at z=30 mm
    phis = [0]
    fnums = [1.5]
    depths = [40]   # 20 in fluid, 40 on interface, 60 in solid
    # 2) call your new compute_fmc_with_interface(...)
    tof_us, FMC = compute_fmc_with_interface(
        M=16, pitch=0.5, b=0.25, f=5e6, 
        c_fluid=1500, c_solid=3000, z_interface=30.0,
        phis=phis, fnums=fnums, z_mm=np.array(depths)
    )
    # 3) backproject and look for the scatterer peak at ~60 mm
    x_im = np.zeros(1); z_im = np.array(depths)
    img = tfm_backprojection(16, 0.5, x_im, z_im, tof_us, FMC, None)
    # assert that the maximum intensity occurs at the known scatterer depth
    max_idx = np.argmax(img[:,0])
    assert abs(z_im[max_idx] - 60.0) < 1e-6

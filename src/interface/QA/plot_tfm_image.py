import os, pandas as pd, matplotlib.pyplot as plt

"""
Load the CSV TFM image and show the 2-D dB map.
"""
script_name = os.path.splitext(os.path.basename(__file__))[0].split('_')[-1]
save_dir    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../results", script_name))

x        = pd.read_csv(os.path.join(save_dir, "x_grid.csv"))["x (mm)"].values
z        = pd.read_csv(os.path.join(save_dir, "z_grid.csv"))["z (mm)"].values
tfm_img  = pd.read_csv(os.path.join(save_dir, "tfm_image.csv"), header=None).values

tfm_db = 20*np.log10(np.abs(tfm_img)+1e-6)

plt.figure(figsize=(8,5))
plt.imshow(tfm_db, extent=[x.min(),x.max(), z.max(),z.min()],
           cmap='hot', aspect='auto', vmin=-40, vmax=0)
plt.colorbar(label='Amplitude (dB)')
plt.scatter([50],[30], c='cyan', marker='x', label='Flaw @ (50,30) mm')
plt.title("TFM Image (dB)")
plt.xlabel("Lateral (mm)")
plt.ylabel("Depth (mm)")
plt.legend()
plt.tight_layout()
plt.show()

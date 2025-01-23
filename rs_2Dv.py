import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1

# Define the function rs_2Dv
def rs_2Dv(b, f, c, e, x, z, Nopt=None):
    # Compute wave number
    kb = 2000 * np.pi * b * f / c
    
    # Determine number of segments
    if Nopt is not None:
        N = Nopt
    else:
        N = round(20000 * f * b / c)
        if N <= 1:
            N = 1
    
    # Use normalized positions in the fluid
    xb = x / b
    zb = z / b
    eb = e / b
    
    # Compute normalized centroid locations for the segments
    xc = np.zeros(N)
    for jj in range(N):
        xc[jj] = -1 + 2 * (jj + 0.5) / N
    
    # Calculate normalized pressure as a sum over all the segments
    p = 0
    for kk in range(N):
        rb = np.sqrt((xb - xc[kk] - eb)**2 + zb**2)
        p += hankel1(0, kb * rb)
    
    # Include external factor
    p = p * (kb / N)
    return p

# Parameters for the first scenario
x = 0
z = np.linspace(5, 200, 500)
f = 5
c = 1500
b = 6.35 / 2

# Compute normalized pressure
p = rs_2Dv(b, f, c, 0, x, z)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(z, np.abs(p))
plt.xlabel('z (mm)')
plt.ylabel('Normalized Pressure |p|')
plt.title('Normalized Pressure vs. z')
plt.grid(True)
plt.show()


# Parameters for the second scenario
x = np.linspace(-10, 10, 200)
z = np.linspace(1, 20, 200)
xx, zz = np.meshgrid(x, z)
f = 5
c = 1500
b = 1

# Compute normalized pressure for the grid
p = rs_2Dv(b, f, c, 0, xx, zz)

# Plot the result as an image
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(p), extent=[x.min(), x.max(), z.min(), z.max()], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Normalized Pressure |p|')
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.title('Normalized Pressure Field |p|')
plt.show()

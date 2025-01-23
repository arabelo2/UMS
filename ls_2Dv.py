import numpy as np
import matplotlib.pyplot as plt

# Define the ls_2Dv function
def ls_2Dv(b, f, c, e, x, z, Nopt=None):
    # Compute wave number
    kb = 2000 * np.pi * b * f / c

    # Determine the number of segments
    if Nopt is not None:
        N = Nopt
    else:
        N = round(20000 * f * b / c)  # Default segment size = 1/10 wavelength
        if N < 1:
            N = 1

    # Use normalized positions in the fluid
    xb = x / b
    zb = z / b
    eb = e / b

    # Compute normalized centroid locations for the segments
    xc = np.zeros(N)
    for jj in range(N):
        xc[jj] = -1 + 2 * (jj + 0.5) / N

    # Calculate normalized pressure
    p = 0
    for kk in range(N):
        ang = np.arctan((xb - xc[kk] - eb) / zb)
        ang = ang + np.finfo(float).eps * (ang == 0)  # Avoid division by zero
        dir_factor = np.sin(kb * np.sin(ang) / N) / (kb * np.sin(ang) / N)
        rb = np.sqrt((xb - xc[kk] - eb)**2 + zb**2)
        ph = np.exp(1j * kb * rb)
        p += dir_factor * np.exp(1j * kb * rb) / np.sqrt(rb)

    # Include external factor
    p = p * (np.sqrt(2 * kb / (1j * np.pi))) / N
    return p

# Common parameters
b = 3  # Half-length of the source (mm)
f = 5  # Frequency (MHz)
c = 1500  # Wave speed (m/s)
x = 0  # On-axis x-coordinate
z = np.linspace(5, 80, 200)  # z-coordinates

# Default segment size (1/10 wavelength)
p_default = ls_2Dv(b, f, c, 0, x, z)

# Using 20 segments (segment size = one wavelength)
p_20_segments = ls_2Dv(b, f, c, 0, x, z, 20)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(z, np.abs(p_default), label='Default Segment Size (1/10 λ)', linewidth=2)
plt.plot(z, np.abs(p_20_segments), label='20 Segments (1 λ each)', linestyle='--', linewidth=2)
plt.xlabel('z (mm)')
plt.ylabel('Normalized Pressure |p|')
plt.title('On-Axis Pressure for Different Segment Sizes')
plt.legend()
plt.grid(True)
plt.show()

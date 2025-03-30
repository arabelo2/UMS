# **Linear MLS Array Modeling (mls_array_modeling)**

## 1. Introduction

This module simulates beamforming using a **linear multi-element source (MLS)** array. It models both **steering** and **focusing** behaviors of 1-D ultrasonic phased arrays, using concepts from *Fundamentals of Ultrasonic Phased Arrays* by **Lester W. Schmerr Jr.**, especially:

- **Chapter 4** – Phased Array Beam Modeling
  - Section **4.5**: Array Beam Modeling Examples
  - Section **4.7**: Beam Steering and Focusing Through a Planar Interface
- **Appendix C.3** – Beam Models for Arrays
- **Appendix C.4** – Miscellaneous Functions
- **Code Listing C.14 and C.15** – Beam modeling and array element geometry

This program uses individual **element models**, **delay laws**, and **windowing functions** to calculate the final acoustic pressure field.

## 2. Mathematical Overview

### 2.1 Element Locations

The location of each element is determined as:

$$
e_m = s \cdot \left(m - 1 - \frac{M - 1}{2}\right), \quad m = 1, 2, ..., M
$$

Where:

- $M$ is the number of elements
- $s = d + g$ is the element **pitch** (element width $d$ plus gap $g$)

### 2.2 Phase Delays

Each element is excited with a complex exponential phase delay:

$$
D_m = e^{j 2\pi f \tau_m}
$$

Where:

- $f$ is the frequency in MHz
- $\tau_m$ is the time delay from the `delay_laws2D` model

### 2.3 Window Amplitudes

Each element is weighted by a taper function:

$$
C_m = w(m)
$$

Where $w(m)$ is a discrete amplitude window (e.g., Rectangular, Hamming, Hann).

### 2.4 Pressure Field

The final pressure field is obtained by superimposing the contributions:

$$
p(x, z) = \sum_{m=1}^{M} C_m D_m p_m(x, z)
$$

Where $p_m(x,z)$ is the beam pattern from the $m$-th element calculated using `ls_2Dv`.

## 3. Implementation Notes

This module is implemented using clean architectural layers:

- **Domain**: `mls_array_modeling.py`
- **Application**: `mls_array_modeling_service.py`
- **Interface**: `mls_array_modeling_interface.py`

It calls auxiliary modules:

- `elements.py` (Code Listing C.15)
- `delay_laws2D_service.py` (for delays)
- `discrete_windows_service.py` (for windowing)
- `ls_2Dv_service.py` (for single element response)

## 4. Examples and Visual Results

### **Steered Beam ($\Phi = 0^\circ$, $F=\infty$)**

```bash
python mls_array_modeling_interface.py --Phi 0 --F inf --M 32 --f 5 --wtype rect
```

![MLS_steered_beam_phi0_Finf_M32_f5_wtyperect](../../examples/figures/MLS_steered_beam_phi0_Finf_M32_f5_wtyperect.png)
This shows a broad beam along the z-axis without convergence. Sidelobes are visible due to the rectangular window.

### **Focused Beam ($\Phi = 0^\circ$, $F=30$ mm)**

```bash
python mls_array_modeling_interface.py --Phi 0 --F 30 --M 32 --f 5 --wtype Han
```

![MLS_steered_focused_beam_phi0_F30_M32_f5_wtypehan](../../examples/figures/MLS_steered_focused_beam_phi0_F30_M32_f5_wtypehan.png)
Here, the beam converges at $z=30$ mm. The Hann window reduces sidelobes and improves focus quality.

### **Steered + Focused Beam ($\Phi = 20^\circ$, $F=30$ mm)**

```bash
python mls_array_modeling_interface.py --Phi 20 --F 30 --M 32 --f 5 --wtype rect
```

![MLS_steered_focused_beam_phi20_F30_M32_f5_wtyperect](../../examples/figures/MLS_steered_focused_beam_phi20_F30_M32_f5_wtyperect.png)
The beam is tilted and focused. This illustrates combined directional and spatial control.

### **Steered + Focused, Higher M**

```bash
python mls_array_modeling_interface.py --Phi 0 --F 30 --M 256 --f 5 --wtype rect
```

![MLS_steered_focused_beam_phi0_F30_M256_f5_wtyperect](../../examples/figures/MLS_steered_focused_beam_phi0_F30_M256_f5_wtyperect.png)
A much narrower main lobe is formed due to the increased number of elements, enhancing resolution.

### **Steered + Focused, $\Phi=15^\circ$, $f=2.5$ MHz**

```bash
python mls_array_modeling_interface.py --Phi 15 --F 30 --M 64 --f 2.5 --wtype Ham
```

![MLS_steered_focused_beam_phi15_F30_M64_f2.5_wtypeham](../../examples/figures/MLS_steered_focused_beam_phi15_F30_M64_f2.5_wtypeham.png)
This lower frequency and moderate element count lead to a wider main lobe with reduced resolution.

## 5. Conclusion

This modeling confirms the theory presented in Schmerr’s text:

- **Beam steering** ($F=\infty$) shifts the wavefront’s direction without convergence. Useful in applications like sector scanning.
- **Focusing** (finite $F$) increases lateral resolution by converging energy at a point.
- **Combined steering + focusing** offers directional control and spatial selectivity — critical for nondestructive evaluation and imaging.

We observe that:

- **Higher M** increases directivity and narrows the beam.
- **Windowing functions** (e.g., Hann vs. Rectangular) strongly affect sidelobe behavior — Hann provides smoother beam profiles.
- **Lower frequency** (e.g., 2.5 MHz) results in broader mainlobes due to longer wavelengths.
- **Focused and steered beams** offer precise beam shaping capabilities — with trade-offs in resolution and beamwidth.

This model serves as a fundamental building block for more advanced simulations, including MPS (Multi-Point Source) and full aperture imaging.

## References

- Schmerr, L. W. (2015). *Fundamentals of Ultrasonic Phased Arrays*. Springer International Publishing.

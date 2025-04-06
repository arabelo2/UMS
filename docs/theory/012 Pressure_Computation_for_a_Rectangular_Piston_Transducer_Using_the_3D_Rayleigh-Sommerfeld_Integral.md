# **Normalized Pressure Computation for a Rectangular Piston Transducer Using the 3D Rayleigh-Sommerfeld Integral**

## 1. Introduction

The **ps_3Dv** module computes the normalized pressure field radiated by a rectangular piston transducer using the 3D Rayleigh-Sommerfeld integral. This simulation is essential for modeling the ultrasound pressure field in applications such as nondestructive evaluation and medical imaging, where phased array transducers are commonly used.

The theoretical basis for this implementation is derived from *Fundamentals of Ultrasonic Phased Arrays* by L.W. Schmerr Jr., particularly from sections discussing:

- The formulation of the Rayleigh-Sommerfeld integral for piston sources,
- Numerical integration over discrete sub-elements,
- The role of directivity and phase in pressure field computation.

The module takes into account the element dimensions, frequency, wave speed, lateral offsets, and evaluation coordinates to produce a complex pressure field. It supports 1D, 2D, and 3D evaluations through a flexible command-line interface.

## 2. Mathematical Formulation

The pressure field is computed by approximating the Rayleigh-Sommerfeld integral over the surface of a rectangular piston. Key equations and their dependencies are as follows:

### 2.1 Wave Number Calculation

```latex
k = \frac{2000 \pi f}{c}
```

- **Dependencies:**
  - `f`: Frequency in MHz  
  - `c`: Wave speed in m/s  
  - The constant `2000\pi` arises from unit conversion (MHz to Hz and m to mm).

### 2.2 Segmentation of the Piston Surface

```latex
P = \left\lceil \frac{1000 f l_x}{c} \right\rceil, \quad Q = \left\lceil \frac{1000 f l_y}{c} \right\rceil
```

- **Dependencies:**  
  - `l_x`, `l_y`: Element dimensions in mm  
  - `f` and `c` for the conversion factor.

### 2.3 Sub-element Centroid Calculation

```latex
x_c^{(p)} = -\frac{l_x}{2} + \frac{l_x}{P}\left(p - \frac{1}{2}\right), \quad y_c^{(q)} = -\frac{l_y}{2} + \frac{l_y}{Q}\left(q - \frac{1}{2}\right)
```

- **Dependencies:**  
  - Element dimensions `l_x`, `l_y`  
  - Number of segments `P`, `Q`

### 2.4 Pressure Field Evaluation

1. **Distance Calculation:**

```latex
r_{pq} = \sqrt{(x - x_c^{(p)} - e_x)^2 + (y - y_c^{(q)} - e_y)^2 + z^2}
```

2. **Directivity (Sinc) Functions:**

```latex
\text{arg}_x = \frac{k\, u_x\, l_x}{2P}, \quad D_x = \frac{\sin(\text{arg}_x)}{\text{arg}_x}
```

```latex
\text{arg}_y = \frac{k\, u_y\, l_y}{2Q}, \quad D_y = \frac{\sin(\text{arg}_y)}{\text{arg}_y}
```

```latex
u_x = \frac{x - x_c^{(p)} - e_x}{r_{pq}}, \quad u_y = \frac{y - y_c^{(q)} - e_y}{r_{pq}}
```

3. **Sub-element Pressure Contribution:**

```latex
p_{pq} = D_x \cdot D_y \cdot \frac{e^{i k r_{pq}}}{r_{pq}}
```

4. **Overall Pressure Calculation:**

```latex
\text{factor} = \frac{-i k (\frac{l_x}{P})(\frac{l_y}{Q})}{2\pi}, \quad p = \text{factor} \cdot \sum_{p,q} p_{pq}
```

## 3. Implementation Details

The **ps_3Dv** model is structured as follows:

- `domain/ps_3Dv.py`: `RectangularPiston3D` class to compute pressure field.
- `application/ps_3Dv_service.py`: service interface to run the computation.
- `src/interface/ps_3Dv_interface.py`: CLI for argument parsing, simulation, and plotting.

### Example CLI Usage

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x=0 --y=0 --z="5,301,10000"
```

## 4. Wavelength and Resolution Considerations

```latex
\lambda = \frac{c}{f} = \frac{1480}{5 \times 10^6} = 2.96 \times 10^{-4}~\text{m} = 0.296~\text{mm}
```

```latex
\text{Resolution} = \frac{\lambda}{10} = 0.0296~\text{mm}
```

This resolution was applied in the examples for spatial sampling along the lateral directions.

## 5. Simulation Results and Figures

### 1D Simulation
```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x=0 --y=0 --z="5,301,10000"
```
<!-- ![1D](../../examples/figures/1D_RS_simulation_for_rectangular_piston_transducer.png) -->

### 2D Simulations
```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y="-5.5204,5.5204,373" --z=50
```
<!-- ![2D z=50](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_z50.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y="-5.5204,5.5204,373" --z=243.243
```
<!-- ![2D Near-Field](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_znear-field_regime.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y="-5.5204,5.5204,373" --z=301
```
<!-- ![2D z=301](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_z301.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y="-5.5204,5.5204,373" --z=10
```
<!-- ![2D z=10](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_z10.png) -->

### 3D Simulations
```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x=0 --y="-5.5204,5.5204,373" --z="5,301,1000"
```
<!-- ![2D x=0](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_x0.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y=0 --z="5,301,1000" --plot-3dfield
```
<!-- ![3D y=0](../../examples/figures/3D_Visualization_of_Ultrasound_Pressure_Field_for_rectangular_piston_transducer_y0.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x=0 --y="-5.5204,5.5204,373" --z="5,301,1000" --plot-3dfield
```
<!-- ![3D x=0](../../examples/figures/3D_Visualization_of_Ultrasound_Pressure_Field_for_rectangular_piston_transducer_x0.png) -->

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x="-5.5204,5.5204,373" --y="-5.5204,5.5204,373" --z=50 --plot-3dfield
```
<!-- ![3D z=50](../../examples/3D_Visualization_of_Ultrasound_Pressure_Field_for_rectangular_piston_transducer_z50.png) -->

## 6. Conclusion

- The wavelength used was `0.296 mm` and the spatial resolution along lateral axes was `0.0296 mm`, ensuring fine sampling.
- Near-field and far-field patterns were distinguished based on theoretical far-field boundary: `z_far ≈ 243 mm`.
- CLI options provide rich control over geometry, frequency, and plotting behavior.
- The figures show expected beam spread, focusing, and directivity effects depending on `z`.

## References

- Schmerr, L. W. (2015). *Fundamentals of Ultrasonic Phased Arrays*. Springer International Publishing.

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

The wave number $\( k \)$ is given by:

$$
k = \frac{2000 \pi f}{c}
$$

- **Dependencies:**  

  - $\( f \)$: Frequency in MHz  
  - $\( c \)$: Wave speed in m/s  
  - The constant $\(2000\pi\)$ arises from unit conversion (MHz to Hz and m to mm).

### 2.2 Segmentation of the Piston Surface

To numerically integrate the pressure, the piston is subdivided into $\( P \)$ segments in the x-direction and $\( Q \)$ segments in the y-direction. If these are not provided, they are computed as:

$$
P = \left\lceil \frac{1000 f l_x}{c} \right\rceil, \quad Q = \left\lceil \frac{1000 f l_y}{c} \right\rceil
$$

- **Dependencies:**  
  - $\( l_x, l_y \)$: Element dimensions in mm  
  - $\( f \)$ and $\( c \)$ for the conversion factor.

### 2.3 Sub-element Centroid Calculation

The centroid of each sub-element is computed using:

$$
x_c^{(p)} = -\frac{l_x}{2} + \frac{l_x}{P}\left(p - \frac{1}{2}\right), \quad p = 1,2,\dots,P
$$

$$
y_c^{(q)} = -\frac{l_y}{2} + \frac{l_y}{Q}\left(q - \frac{1}{2}\right), \quad q = 1,2,\dots,Q
$$

- **Dependencies:**  
  - Element dimensions $\( l_x, l_y \)$ 
  - Number of segments $\( P, Q \)$

### 2.4 Pressure Field Evaluation

For each sub-element, the contribution to the pressure at an evaluation point $\((x,y,z)\)$ is computed as:

1. **Distance Calculation:**

   $$
   r_{pq} = \sqrt{\left(x - x_c^{(p)} - e_x\right)^2 + \left(y - y_c^{(q)} - e_y\right)^2 + z^2}
   $$

   - **Dependencies:**

     - Evaluation coordinates $\(x, y, z\)$ (in mm)  
     - Sub-element centroids $\(x_c^{(p)}, y_c^{(q)}\)$
     - Lateral offsets $\( e_x, e_y \)$

2. **Directivity (Sinc) Functions:**

   The directivity factors in x and y directions are given by:

   $$
   \text{arg}_x = \frac{k\, u_x\, l_x}{2P}, \quad D_x = \frac{\sin(\text{arg}_x)}{\text{arg}_x}
   $$

   $$
   \text{arg}_y = \frac{k\, u_y\, l_y}{2Q}, \quad D_y = \frac{\sin(\text{arg}_y)}{\text{arg}_y}
   $$

   where the direction cosines are:

   $$
   u_x = \frac{x - x_c^{(p)} - e_x}{r_{pq}}, \quad u_y = \frac{y - y_c^{(q)} - e_y}{r_{pq}}.
   $$

   - **Dependencies:**

     - Wave number $\( k \)$
     - Element dimensions $\( l_x, l_y \)$ and segmentation $\( P, Q \)$
     - Direction cosines $\( u_x, u_y \)$

3. **Sub-element Pressure Contribution:**

   The contribution from each sub-element is:

   $$
   p_{pq} = D_x \, D_y \, \frac{e^{i k\, r_{pq}}}{r_{pq}}
   $$

4. **Overall Pressure Calculation:**

   After summing the contributions from all sub-elements, the pressure is scaled by:

   $$
   \text{factor} = \frac{-i k \left(\frac{l_x}{P}\right) \left(\frac{l_y}{Q}\right)}{2\pi}
   $$

   So that the total pressure is:

   $$
   p = \text{factor} \sum_{p=1}^{P} \sum_{q=1}^{Q} p_{pq}
   $$

   - **Dependencies:**
  
     - Integration over sub-elements  
     - Element geometry, frequency, wave speed, and offsets

## 3. Implementation Details

The **ps_3Dv** model is structured as follows:

- **Domain Layer:**

  The module `domain/ps_3Dv.py` defines the `RectangularPiston3D` class that encapsulates the computation of the pressure field. This class includes:
  - The computation of the wave number $\( k \)$
  - Determination of integration parameters $\( P \)$ and $\( Q \)$
  - Calculation of sub-element centroids and subsequent pressure contributions via the Rayleigh-Sommerfeld integral

- **Service Layer:**

  The service function `run_ps_3Dv_service` (in `application/ps_3Dv_service.py`) provides an abstraction layer that instantiates the `RectangularPiston3D` class and calls its pressure computation method.

- **Interface Layer:**
 
  The CLI module `src/interface/ps_3Dv_interface.py` handles:
  - Parsing of command-line arguments (element dimensions, frequency, wave speed, evaluation coordinates, optional integration parameters)
  - Broadcasting and meshgrid creation for 1D, 2D, or 3D simulations
  - Plotting of the computed pressure field using matplotlib with consistent styling via the helper function `apply_plot_style`.

### Example CLI Usage

```bash
python src/interface/ps_3Dv_interface.py --lx=6 --ly=12 --f=5 --c=1480 --ex=0 --ey=0 --x=0 --y=0 --z="5,123.4,4000"
```

- This command computes a 1D pressure field (x and y scalars, z vector) over the specified range.

## 4. Simulation Results and Visualization

The simulation outputs can be visualized using the provided CLI interface. The figures below represent different simulation configurations. (All image links are commented out; replace the paths with actual image paths as needed.)

<!-- 
![1D Simulation](../../examples/figures/1D_RS_simulation_for_rectangular_piston_transducer.png)
-->

<!-- 
![2D Simulation at z=50 mm](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_z50.png)
-->

<!-- 
![2D Simulation in Near-Field Regime](../../examples/figures/2D_RS_simulation_for_rectangular_piston_transducer_znear‑field_regime.png)
-->

<!-- 
![3D Visualization for y=0](../../examples/figures/3D_Visualization_of_Ultrasound_Pressure_Field_for_rectangular_piston_transducer_y0.png)
-->

These images illustrate:

- **1D Simulation:**  
  The pressure field along a single dimension showing the phase and amplitude variation predicted by the Rayleigh-Sommerfeld integral.

- **2D Simulation:**  
  Pressure field maps (using `imshow` or 3D scatter plots) for various fixed coordinate cases (e.g., at \(z = 50\,\text{mm}\) or in near-field versus far-field regimes). The resolution is determined by the wavelength and integration parameters.

- **3D Field Visualization:**  
  3D scatter plots display the ultrasound pressure field in a volumetric representation, highlighting the spatial variations.

## 5. Comparative Analysis

Based on our simulation and theoretical understanding:

- **Near-Field vs. Far-Field:**  
  Using the far-field criterion

  $$
  r \gg \frac{2b^2}{\lambda}
  $$

  (with $\(b\)$ chosen as half the larger element dimension), we determined that for typical parameters (e.g., $\(l_y = 12\,\text{mm}\)$, so $\(b=6\,\text{mm}\)$ and $\(\lambda \approx 0.296\,\text{mm}\)$), the far-field threshold is approximately $\(243\,\text{mm}\)$. Our simulation ranges (e.g., $\(z\)$ up to 4000 mm) may span both regimes. Visual inspection of the pressure field indicates differences in beam collimation and directivity between the near-field and far-field.

- **Directivity Effects:**  
  The directivity factors (sinc functions) modulate the contribution of each sub-element. Their proper evaluation ensures that the angular dependence of the radiated pressure field is accurately captured.

- **Integration Resolution:**  
  The number of segments $\( P \)$ and $\( Q \)$ (determined automatically or provided by the user) directly affect the numerical accuracy. Higher resolution is required for larger elements or when simulating very near-field phenomena.

## 6. Conclusion

The **ps_3Dv** module implements a numerical approximation of the 3D Rayleigh-Sommerfeld integral to compute the normalized pressure field of a rectangular piston transducer. Our approach is based on segmenting the transducer surface and summing the contributions from each sub-element while accounting for directivity and phase delay.

**Key findings include:**

- **Theoretical Consistency:**  
  The derived equations—from the computation of the wave number $\( k \)$ to the integration over sub-elements—are consistent with the theory presented in *Fundamentals of Ultrasonic Phased Arrays* (see Sections 6 and 7 for beam modeling).

- **Dependence on Parameters:**  
  The simulation accurately reflects the dependencies on element dimensions, frequency, wave speed, lateral offsets, and integration resolution. This allows for flexible simulation of near-field and far-field regimes.

- **Visualization and Validation:**  
  The output figures (1D, 2D, and 3D visualizations) validate the expected spatial behavior of the pressure field. In the near-field, the pressure exhibits rapid variations with distance, while in the far-field, the field becomes smoother and more directional.

Overall, the simulation results support the theoretical predictions, providing a robust tool for exploring ultrasound pressure fields in rectangular piston transducers.

## References

- Schmerr, L. W. (2015). *Fundamentals of Ultrasonic Phased Arrays*. Springer International Publishing.

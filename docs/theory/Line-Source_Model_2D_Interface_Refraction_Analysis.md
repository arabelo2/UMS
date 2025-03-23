# **Theoretical and Numerical Analysis of Acoustic Fields Through a Fluid-Fluid Interface Using the Line Source Model**

## 1. Introduction

In ultrasonic testing and modeling, predicting the pressure field transmitted across interfaces between different media is crucial for accurate simulations. The Line Source Model (LS model), as described in *Fundamentals of Ultrasonic Phased Arrays* by Lester W. Schmerr Jr., provides an efficient framework to analyze these pressure fields. This section explores the LS model for simulating acoustic wave propagation through a fluid-fluid interface, based on Chapters **2.5**, **4.7**, and Appendix **C.5** of the book.

The analysis is implemented through the `ls_2Dint_interface.py` module, which evaluates the normalized pressure field produced by a 2-D piston source at an oblique angle to the interface. The model incorporates ray theory principles and accounts for refraction effects occurring at the boundary between two fluids.

## 2. Theoretical Model: LS2D with Interface Refraction

The pressure field at any point in the second fluid is computed using a Rayleigh-Sommerfeld type integral modified by ray theory. The model discretizes the 2-D source into multiple line segments, each treated independently to capture near-field interference and refraction effects. According to Schmerr (2015), this method is especially relevant for simulations involving planar interfaces and steered beams.

### Governing Equations and Variables

The normalized pressure field \( p(x, z) \) is given by:

$$
p(x, z) = \frac{\sqrt{2 k_1 b / (j\pi)}}{N} \sum_{j=1}^{N} T_p(\theta_{j}) \cdot D(\theta_{j}) \cdot \frac{e^{i(k_1 r_1 + k_2 r_2)}}{\sqrt{r_1 + (c_2/c_1)r_2 \cdot \frac{\cos^2(\theta_1)}{\cos^2(\theta_2)}}}
$$

Where:

- \( b \): Half-length of the piston source (mm)
- \( f \): Frequency (MHz)
- \( c_1, c_2 \): Sound speeds in the two fluids (m/s)
- \( d_1, d_2 \): Densities in the two fluids (g/cm\(^3\))
- \( k_1 = 2000 \pi b f / c_1 \), \( k_2 = 2000 \pi b f / c_2 \): Normalized wave numbers
- \( T_p \): Pressure transmission coefficient
- \( D \): Directivity function
- \( r_1 \), \( r_2 \): Normalized path lengths through the two media

### Segment Discretization

The number of segments \( N \) used to discretize the source is either user-defined via the `--Nopt` argument or automatically calculated using:

$$
N = \text{round}\left( \frac{2000 f b}{c_1} \right)
$$

This ensures that each segment is no larger than one wavelength in medium 1. For \( N = 1 \), the source is modeled as a single radiator; for \( N > 1 \), composite source behavior is captured.

## 3. CLI Commands and Simulation Setup

Simulations were conducted using the CLI interface provided by `ls_2Dint_interface.py`. The following commands generated the visualizations discussed:

### **2-D Simulation with Automatically Computed N**

```sh
python src/interface/ls_2Dint_interface.py \
    --b 3 --f 5 --c 1480 \
    --mat "1,1480,7.9,5900" --e 0 \
    --angt 10.217 --Dt0 50.8 \
    --x2="0,29.6,1000" --z2="1,30.6,1000"
```

![Auto-computed N](../../examples/figures/Line-Source_Model_2-D_piston_fluid-fluid_Nauto.png)

- **Lambda** = \( c / f = 1480 / 5\times10^6 = 0.296 \text{ mm} \)
- **Resolution** = \( \lambda / 10 = 0.0296 \text{ mm} \)

### **2-D Simulation with N = 15**

```sh
python src/interface/ls_2Dint_interface.py \
    --b 3 --f 5 --c 1480 \
    --mat "1,1480,7.9,5900" --e 0 \
    --angt 10.217 --Dt0 50.8 \
    --x2="0,29.6,1000" --z2="1,30.6,1000" --Nopt 15
```

![Manual N=15](../../examples/figures/Line-Source_Model_2-D_piston_fluid-fluid_N15.png)

## 4. Analysis and Observations

The normalized pressure fields show expected beam steering and refraction at the fluid-fluid interface.

- **Automatically Computed N**: Exhibits a smooth and continuous beam path with adequate resolution.
- **N = 15**: Reveals sharper features and more prominent interference lobes.

Both simulations confirm beam refraction toward the slower medium (Snell's Law), as expected. The steered beam behavior aligns with theoretical expectations detailed in **Sections 2.5** and **4.7**.

## 5. Conclusion

The LS2D interface model is a powerful approach for simulating pressure fields across fluid-fluid boundaries. It combines asymptotic approximations with ray theory to provide a computationally efficient yet physically realistic solution. The use of segment-based discretization and parameter tuning (e.g., \( N \)) offers flexibility and precision. The results validate theoretical predictions from Schmerr (2015), confirming the model's utility in ultrasonic phased array simulations.

## References

Schmerr, L. W. (2015). *Fundamentals of Ultrasonic Phased Arrays*. Springer International Publishing.

- Section 2.5: Radiation Through a Planar Interface
- Section 4.7: Beam Steering and Focusing through a Planar Interface
- Appendix C.5: Code Listings

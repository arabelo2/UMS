# **Ultrasonic Measurements Simulator (UMS)**

UMS is an advanced Ultrasonic Measurements Simulator built using Object-Oriented Programming (OOP), Clean Architecture, and Clean Code principles. Its modular design—divided into domain, application, and interface layers—ensures maintainability, scalability, and ease of extension. The project bases its methodologies on the theories presented in Schmerr’s _Fundamentals of Ultrasonic Phased Arrays_ (2014) and other supporting literature available in the repository.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Design](#architecture--design)
  - [Code Dependency Tree](#code-dependency-tree)
  - [Theory Document Tree](#theory-document-tree)
- [Modules](#modules)
  - [Fresnel 2D](#fresnel-2d)
  - [Gaussian Beam 2D](#gaussian-beam-2d)
  - [Non-Paraxial Gaussian 2D](#non-paraxial-gaussian-2d)
  - [Delay Laws 2D](#delay-laws-2d)
  - [Discrete Windows](#discrete-windows)
  - [Elements](#elements)
  - [MLS Array Modeling at Fluid/Fluid Interface](#mls-array-modeling-at-fluidfluid-interface)
  - [Delay Laws 2D for Interface](#delay-laws-2d-for-interface)
- [Usage Instructions](#usage-instructions)
- [Development Guidelines](#development-guidelines)
- [References](#references)

---

## Overview

UMS is designed to simulate ultrasonic pressure fields using various methods such as Fresnel integration, Gaussian beam approximations, and advanced delay law computations. The project is structured into a well-defined layered architecture:
  
- **Domain Layer:** Contains core computational logic and implements the mathematical foundations (e.g., integrals, beam models, and delay laws).  
- **Application Layer:** Manages input validation, data handling, and orchestrates calls to domain logic.  
- **Interface Layer:** Provides command-line interfaces (CLIs) for invoking simulations, visualizing results, and managing output.

The theoretical foundations for these implementations are documented in the [`docs/theory`](../docs/theory/) folder.

---

## Architecture & Design

### Code Dependency Tree

The **src** folder implements the main business logic of the project. Below is a simplified tree diagram showing the dependency relationships among its subdirectories:

```
src/
├── application/
│   ├── fresnel_2D_service.py
│   ├── gauss_2D_service.py
│   ├── np_gauss_2D_service.py
│   ├── delay_laws2D_service.py
│   ├── mls_array_model_int_service.py
│   └── delay_laws2D_int_service.py
├── domain/
│   ├── fresnel_2D.py
│   ├── on_axis_foc2D.py
│   ├── gauss_2D.py
│   ├── np_gauss_2D.py
│   ├── delay_laws2D.py
│   ├── discrete_windows.py
│   ├── elements.py
│   ├── mls_array_model_int.py
│   └── delay_laws2D_int.py
└── interface/
    ├── fresnel_2D_interface.py
    ├── on_axis_foc2D_interface.py
    ├── gauss_2D_interface.py
    ├── np_gauss_2D_interface.py
    ├── delay_laws2D_interface.py
    ├── mls_array_model_int_interface.py
    └── delay_laws2D_int_interface.py
```

> **Note:** The layered design enforces a clear separation of concerns: **Interface → Application → Domain**. Each CLI interface calls a corresponding service in the Application layer, which in turn executes the core logic from the Domain layer.

### Theory Document Tree

The theoretical principles behind the implementations are documented in the [`docs/theory`](../docs/theory/) folder. A simplified diagram of its structure is as follows:

```
docs/theory/
├── Fresnel_Integral.md
├── Gaussian_Beam_Theory.md
├── Non_Paraxial_Gaussian.md
├── Delay_Laws.md
└── Window_Functions.md
```

> **Mapping:**  

> - **Fresnel_Integral.md:** Supports the Fresnel 2D module.  
> - **Gaussian_Beam_Theory.md:** Provides details for the Gaussian Beam 2D module.  
> - **Non_Paraxial_Gaussian.md:** Details the model behind Non-Paraxial Gaussian 2D.  
> - **Delay_Laws.md:** Explains the derivation of delay laws used in both steering/focusing modules.  
> - **Window_Functions.md:** Contains theory behind the discrete windowing functions.

---

## Modules

Each simulation module in UMS has a clear division into three layers. Here is an overview of the primary modules:

### Fresnel 2D

- **Purpose:** Computes the normalized pressure field for a focused piston transducer using the Fresnel integral approximation.
- **Layers:**
  - *Domain:* `fresnel_2D.py`
  - *Application:* `fresnel_2D_service.py`
  - *Interface:* `fresnel_2D_interface.py`
- **Reference:** See [`Fresnel_Integral.md`](../docs/theory/004%20Fresnel_Integral_Model_Lateral_and_On_Axis_Pressure_Field_Simulation.md) for theoretical background.

### Gaussian Beam 2D

- **Purpose:** Simulates the Gaussian beam pressure field using a 15-coefficient model.
- **Layers:**
  - *Domain:* `gauss_2D.py`
  - *Application:* `gauss_2D_service.py`
  - *Interface:* `gauss_2D_interface.py`
- **Reference:** See [`Gaussian_Beam_Theory.md`](../docs/theory/009%20Gaussian_MLS_Array_Modeling.md).

### Non-Paraxial Gaussian 2D

- **Purpose:** Provides pressure field computations using a non-paraxial model based on a 10-coefficient system.
- **Layers:**
  - *Domain:* `np_gauss_2D.py`
  - *Application:* `np_gauss_2D_service.py`
  - *Interface:* `np_gauss_2D_interface.py`
- **Reference:** See [`Non_Paraxial_Gaussian.md`](../docs/theory/005%20Gauss_Integral_Model_Lateral_Pressure_Field_Simulation.md).

### Delay Laws 2D

- **Purpose:** Computes time delays for an M-element transducer array for both steering and focusing.
- **Layers:**
  - *Domain:* `delay_laws2D.py`
  - *Application:* `delay_laws2D_service.py`
  - *Interface:* `delay_laws2D_interface.py`
- **Reference:** See [`Delay_Laws.md`](../docs/theory/011%20Delay_Laws_1D_Array_2D_Interface_Model.md).

### Discrete Windows

- **Purpose:** Generates various windowing functions (Hanning, Hamming, etc.) for array apodization.
- **Layers:**
  - *Domain:* `discrete_windows.py`
  - *Application:* `discrete_windows_service.py`
  - *Interface:* `discrete_windows_interface.py`
- **Reference:** See [`Window_Functions.md`](../docs/theory/007%20Discrete_Windows.md).

### Elements

- **Purpose:** Calculates array element parameters such as size, gap, total array length, and centroid positions.
- **Layer:** Implemented in `elements.py` (invoked directly by other modules).

### MLS Array Modeling at Fluid/Fluid Interface

- **Purpose:** Simulates the pressure field for an array propagating waves through a fluid/fluid interface.
- **Layers:**
  - *Domain:* `mls_array_model_int.py`
  - *Application:* `mls_array_model_int_service.py`
  - *Interface:* `mls_array_model_int_interface.py`

### Delay Laws 2D for Interface

- **Purpose:** Computes delay laws for arrays across a two-media interface, handling both steering-only and steering-and-focusing cases.
- **Layers:**
  - *Domain:* `delay_laws2D_int.py`
  - *Application:* `delay_laws2D_int_service.py`
  - *Interface:* `delay_laws2D_int_interface.py`

---

## Usage Instructions

Each module comes with its own Command Line Interface (CLI). For example:

- To run the **Fresnel 2D** simulation:

  ```bash
  python src/interface/fresnel_2D_interface.py --options
  ```

- To simulate the **MLS Array Model**:

  ```bash
  python src/interface/mls_array_model_int_interface.py --f 1e6 --d1 20 --c1 1500 --... [other parameters]
  ```
  
Check the help command (`--help`) for detailed usage instructions on each CLI tool.

---

## Development Guidelines

- **Object-Oriented Design:** All modules are implemented using OOP principles to ensure modularity and reusability.
- **Clean Architecture:** The clear separation of domain, application, and interface layers provides a robust, testable codebase.
- **Testing & Validation:** Modules include edge-case handling (e.g., division by zero) and are designed to support unit tests.

---

## References

- Schmerr, L. W. (2014). *Fundamentals of Ultrasonic Phased Arrays*. Springer International Publishing. ISBN: 9783319072722.
- Additional theoretical documents are available in the [`docs/theory`](../docs/theory/) folder.

---

Feel free to contribute, report issues, or suggest improvements. Enjoy simulating ultrasonic measurements with UMS!
```

# UMS
Ultrasonic Measurements Simulator

"These programs are based on the methodologies presented in the book by Schmerr (2014)."

### Reference:

SCHMERR, L. W. Fundamentals of Ultrasonic Phased Arrays. Cham: Springer International Publishing, 2014. (Solid Mechanics and Its Applications). ISBN 9783319072722. Available at: https://books.google.com.br/books?id=_H1HBAAAQBAJ. Accessed on: 25 Jan. 2025.

### **Overview of Programs**

The programs follow **Clean Architecture**, **Object-Oriented Programming (OOP) principles**, and **best coding practices**. They were broken down into **domain, application, and interface layers**, ensuring modularity, maintainability, and scalability.

---

## **1. Fresnel 2D**
### **Purpose:**
Computes the normalized pressure field for a **1D focused piston transducer** using the **Fresnel integral approximation**.

### **Implementation in Python:**
- **Domain:** `fresnel_2D.py` - Implements the mathematical logic for Fresnel integral computation.
- **Application:** `fresnel_2D_service.py` - Calls the domain function and ensures input validity.
- **Interface:** `fresnel_2D_interface.py` - CLI for running the Fresnel 2D computation with plotting support.

### **Key Features:**
✅ Fully object-oriented implementation.  
✅ Applied edge-case handling (e.g., division by zero).  
✅ Integrated **NumPy** and **Matplotlib** for efficient computation and visualization.

---

## **2. On-Axis Focused 2D**
### **Purpose:**
Computes the **on-axis pressure field** of a **focused piston transducer** using **Fresnel integral methods**.

### **Implementation in Python:**
- **Domain:** `on_axis_foc2D.py` - Handles the core computation using the Fresnel integral.
- **Application:** `on_axis_foc2D_service.py` - Calls the domain function and prepares data.
- **Interface:** `on_axis_foc2D_interface.py` - CLI interface allowing users to compute and visualize pressure fields.

### **Key Features:**
✅ Edge-case handling for **z = 0** (division by zero).  
✅ Supports complex number operations for **pressure calculations**.  
✅ Unit-tested with multiple edge cases.

---

## **3. Gaussian Beam 2D**
### **Purpose:**
Computes the **Gaussian beam approximation** of the pressure field for a **1D transducer**.

### **Implementation in Python:**
- **Domain:** `gauss_2D.py` - Uses **Wen & Breazeale's 15-coefficient Gaussian model**.
- **Application:** `gauss_2D_service.py` - Provides input validation and prepares data.
- **Interface:** `gauss_2D_interface.py` - CLI to compute, visualize, and save pressure field results.

### **Key Features:**
✅ Uses **NumPy arrays** for efficient Gaussian calculations.  
✅ Includes **Gauss coefficient retrieval** from `gauss_c15.py`.  
✅ Supports **plotting** and **file-based outputs**.

---

## **4. Non-Paraxial Gaussian 2D**
### **Purpose:**
Computes the **non-paraxial Gaussian beam pressure field** using **Wen & Breazeale’s 10-coefficient model**.

### **Implementation in Python:**
- **Domain:** `np_gauss_2D.py` - Implements the **non-paraxial expansion model**.
- **Application:** `np_gauss_2D_service.py` - Manages input validation and execution.
- **Interface:** `np_gauss_2D_interface.py` - CLI tool for computation, visualization, and output file storage.

### **Key Features:**
✅ Fully modularized using **OOP principles**.  
✅ Retrieves **10 Gaussian coefficients** from `gauss_c10.py`.  
✅ Efficient implementation using **NumPy vectorized operations**.

---

## **5. Delay Laws 2D**
### **Purpose:**
Computes **time delays** for an **M-element transducer array**, supporting **steering-only and steering + focusing** cases.

### **Implementation in Python:**
- **Domain:** `delay_laws2D.py` - Implements mathematical logic for computing **time delays**.
- **Application:** `delay_laws2D_service.py` - Ensures correct function calls and input validation.
- **Interface:** `delay_laws2D_interface.py` - CLI tool for computation and visualization.

### **Key Features:**
✅ Uses **steering angle and focal distance** for accurate delay calculations.  
✅ Handles **infinity (`inf`) cases** for steering-only scenarios.  
✅ Robust error handling for **invalid element configurations**.

---

## **6. Discrete Windows**
### **Purpose:**
Generates **windowing functions** (Hanning, Hamming, Blackman, Triangular, Cosine, and Rectangular) for **array apodization**.

### **Implementation in Python:**
- **Domain:** `discrete_windows.py` - Implements window function calculations.
- **Application:** `discrete_windows_service.py` - Provides an interface for retrieving the amplitude values.
- **Interface:** `discrete_windows_interface.py` - CLI for generating and visualizing window functions.

### **Key Features:**
✅ Supports **six standard apodization methods**.  
✅ Implements **efficient NumPy vectorized operations**.  
✅ Includes **visualization support** via **Matplotlib**.

---

## **7. Elements**
### **Purpose:**
Computes **array element size, gap size, total array length, and centroid locations**.

### **Implementation in Python:**
- **Domain:** `elements.py` - Implements an **OOP-based calculator** for transducer array elements.
- **Application:** (No separate layer needed, called directly from domain).
- **Interface:** (Handled via higher-level modules like `mls_array_modeling.py`).

### **Key Features:**
✅ **Validated input parameters** (frequency, wave speed, element count).  
✅ Uses **OOP encapsulation** for modularity.  
✅ Generates **centroid locations dynamically** for any array size.

---

## **8. MLS Array Modeling**
### **Purpose:**
Simulates **Multi-Line Source (MLS) array beamforming**, integrating **elements, delay laws, windowing, and wave propagation**.

### **Implementation in Python:**
- **Domain:** `mls_array_modeling.py` - Implements **MLS modeling logic** using OOP principles.
- **Application:** `mls_array_modeling_service.py` - Manages function calls and result processing.
- **Interface:** `mls_array_modeling_interface.py` - CLI tool for simulation and visualization.

### **Key Features:**
✅ Uses **delay laws, discrete windows, and wave propagation models**.  
✅ Efficient **NumPy-based computation** for large arrays.  
✅ Generates **heatmaps** of beamforming results using **Matplotlib**.

---

## **Overall Achievements**
✅ **Implemented Clean Architecture (Domain, Application, Interface layers)**  
✅ **Applied OOP principles for modular and reusable code**  
✅ **Optimized computational efficiency with NumPy and SciPy**  
✅ **Handled edge cases (e.g., division by zero, invalid input values)**  
✅ **Automated testing for robustness and reliability** 

---

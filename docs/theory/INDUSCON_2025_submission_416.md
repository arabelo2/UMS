\documentclass[a4paper, 10pt, conference, encapsulated]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{CJK} % use CJK for pdfLaTeX
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\addtolength{\topmargin}{0cm}
\addtolength{\textheight}{0.7in}
\usepackage{xcolor}
\usepackage{newunicodechar}
\usepackage{makecell} % For \makecell
\newunicodechar{∼}{\textasciitilde}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Digital Twin-Driven FMC-TFM Simulation for Ultrasonic Phased Arrays across Fluid-Solid Interfaces: Delay Law and FWHM Analysis}

\author{
\IEEEauthorblockN{Alexandre Rabelo}
\IEEEauthorblockA{
\textit{Dept. of Mechatronics Engineering} \\
\textit{University of São Paulo, USP} \\
São Paulo, Brazil \\
arabelo@usp.br}
\and
\IEEEauthorblockN{Timoteo F. de Oliveira}
\IEEEauthorblockA{
\textit{Dept. of Mechatronics Engineering} \\
\textit{University of São Paulo, USP} \\
São Paulo, Brazil \\
tim@usp.br}
\and
\IEEEauthorblockN{Flávio Buiochi}
\IEEEauthorblockA{
\textit{Dept. of Mechatronics Engineering} \\
\textit{University of São Paulo, USP} \\
São Paulo, Brazil \\
fbuiochi@usp.br}
}

\maketitle

\begin{abstract}

This paper presents a digital twin-driven simulation framework for modeling and analyzing the inspection of ultrasonic phased arrays through fluid-solid interfaces, with a focus on the beamforming principles of FMC and TFM. The system simulates a phased array immersed in a coupling fluid and evaluates beam propagation and focusing behavior across planar boundaries using high-frequency ray theory. The pipeline integrates field simulation and Snell's law-based delay law computation, fully implemented in Python and grounded in canonical models from the literature. Quantitative evaluation is performed using FWHM metrics and delay law resolution analysis. The results confirm the ability of the framework to replicate key focus characteristics and quantify resolution degradation due to delay quantization. Multiple FWHM estimation methods are benchmarked, reinforcing the framework's utility to validate and tune FMC-TFM-based strategies in multilayer media environments.

\end{abstract}

\begin{IEEEkeywords}
Ultrasonic simulation, digital twin, phased array, FMC, TFM, fluid–solid interface, FWHM estimation.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:introduction}

Nondestructive testing (NDT) plays a vital role in the evaluation of structural integrity in industries such as aerospace, power generation, oil and gas. Among various techniques, ultrasonic phased array systems are widely adopted for their beam steering, focusing, and real-time imaging capabilities \cite{b3}, \cite{b6}. However, inspection of components immersed in coupling fluids, such as water, and bonded to solid materials poses a significant challenge due to beam distortion at the fluid–solid interface. Effects such as refraction, transmission losses, and mode conversion often degrade resolution and reduce the detectability of defects at depth \cite{b3}, \cite{b6}.

To address these limitations, full matrix capture (FMC) combined with the total focusing method (TFM) has emerged as a powerful framework for synthetic focus at every point in the region of interest \cite{b6}. In this approach, all pairs of transmit-receive elements are activated to record time-domain signals, enabling offline reconstruction of high-resolution images with improved lateral resolution. The method is particularly suitable for detecting small defects and evaluating components with complex geometries \cite{b2}, \cite{b6}.

Despite advances in FMC-TFM-based imaging, modeling and quantifying the behavior of acoustic beams across interfaces remains a challenge. Many simulation platforms simplify wave propagation, often assuming homogeneous media, and neglect important details like refraction, anisotropy, or focal-law accuracy in interface-transmitted beams. Additionally, the high cost and limited flexibility of the experimental setups constrain the repeatability and systematic study of parameter variation. Recent advances in simulation technology, particularly those informed by the concept of digital twins (DT), offer new opportunities for realistic, tunable, and high-fidelity imaging system design \cite{b2}.

In this work, we propose a simulation pipeline based on the digital twin paradigm to study FMC-TFM-based ultrasonic beamforming through fluid-solid interfaces. The pipeline is grounded in canonical models presented by Schmerr~\cite[Sec.~2.5,4.2,4.3,4.7,5.1,5.2]{b1}, specifically: (i) radiation theory through a planar interface, (ii) array beam steering and focusing formulations, (iii) calculation of time delay laws in multilayer media, and (iv) pressure field modeling using analytical and numerical tools. These physical models were faithfully implemented in Python without altering their theoretical basis. The contribution of this work lies in the architecture and integration of these models into a modular and extensible simulation pipeline that enables parameterized studies of beamforming through fluid–solid interfaces using digital twin principles. Moreover, we incorporated beam quality assessment based on FWHM, applying signal envelope extraction and full-width estimation methods directly inspired by Rainio~\textit{et al.}~\cite{b7}, which were reused without methodological changes. The resulting framework enables reproducible simulations, automated configuration, and comparative beam analysis in realistic interface scenarios.

This paper is structured as follows: Section~\ref{sec:theoretical_background} covers beam modeling and delay law theory. Section~\ref{sec:DT-driven_simulation_framework} details the digital twin simulation framework. Section~\ref{sec:FMC-TFM_imaging_and_quantification_methodology} explains the FMC-TFM imaging workflow. Section~\ref{sec:results_and_discussion} presents and discusses simulation results. Section~\ref{sec:conclusion} concludes and outlines future work.

\section{Theoretical Background}
\label{sec:theoretical_background}

The modeling of ultrasound beams across planar fluid-solid interfaces requires an accurate description of pressure and velocity fields that incorporate reflection, refraction, transmission, and geometric spreading effects. These phenomena arise from solving the scalar wave equation for layered media using high-frequency approximations. In this context, we adopt analytical models based on rays derived from Schmerr's theoretical formulation \cite{b1}.

\subsection{Piston Source Model}

The pressure field radiated by a finite-size transducer element into a homogeneous fluid medium can be modeled using a line-source approximation, discretized into equally spaced segments. This model, described in Schmerr \cite{b1}, evaluates the total field as a superposition of contributions from each segment, represented by zeroth-order Hankel functions.

The normalized pressure field at a field point \((x, z)\) due to a uniform piston source of half width \(b\) is given by the following:

\begin{equation}
\frac{p(x, z, \omega)}{\rho_{1} c_{1} v_0(\omega)} = \frac{k_{1} b}{N} \sum_{n=1}^{N} H_0^{(1)}\left(k_{1} b \bar{r}_n\right),
\label{eq:rs2Dv}
\end{equation}
where \(H_0^{(1)}\) is the zeroth-order Hankel function of the first kind,  
\(k_{1} = \omega / c_{1}\) is the wavenumber in the fluid,  
\(\rho_{1}\) and \(c_{1}\) are the density of the fluid and the wave speed,
\(v_0(\omega)\) is the input velocity spectrum,  
\(N\) is the number of ray segments,  
\(\bar{r}_n = \sqrt{(x / b - \bar{x}_n)^2 + (z / b)^2}\) is the normalized distance from segment \(n\) to the field point,  
and \(\bar{x}_n = -1 + \frac{2}{N} \left(n - \frac{1}{2} \right)\) defines the normalized center location of the \(n\)-th segment.

\subsection{Line Source Model}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=1\linewidth]{Figures-Figure 2.14 Schmerr's.drawio.png}
  \caption[Geometry of the line‑source interface model]{Geometry for implementing a multiple line source model of a 1-D element of an array radiating waves through a planar interface between two different media. The dashed line is the extended array axis, \(C\) marks the center of the subelement, and \(P(x,z)\) is the observation point. All symbols are as defined in the text (adapted from Schmerr~\cite{b1}).}
  \label{fig:interface_geometry}
\end{figure}

The linear source program numerically evaluates the pressure field at a field point \(P(x,z)\) generated by a 1-D linear array source simulated in a 2-D spatial domain that radiates through a planar interface between two different media~\cite{b1}. This model considers the effects of refraction and transmission at the interface between a fluid and a solid, as detailed by Schmerr~\cite{b1}. The total pressure field is obtained by summing the contributions of multiple ray segments across the aperture, as expressed by the following:

\begin{align}
p(x, z) &= \rho_{1} c_{1} v_{0}(\omega) \sqrt{\frac{2k_{1} b}{\pi i}} \notag \\ &\quad \times \frac{1}{N} \sum_{n=1}^{N}\frac{T_{p0}(\theta_{10}^{n}) \exp\left(i k_{1} b\bar{r}_{10}^{n} + i k_{2} b\bar{r}_{20}^{n}\right)}{\sqrt{\bar{r}_{10}^{n} + \frac{c_{2}}{c_{1}} \frac{\cos^2(\theta_{10}^{n})}{\cos^2(\theta_{20}^{n})} \bar{r}_{20}^{n}}} D_{\frac{b}{N}}(\theta),
\label{eq:ls2dint}
\end{align}
where \(\rho_1\) and \(c_1\) are the fluid density and sound speed in medium 1, \(v_0(\omega)\) is the input velocity spectrum, \(k_1 = \omega/c_1\) and \(k_2 = \omega/c_2\) are the wavenumbers in the fluid and solid, \(b\) is the half-aperture width of the array, \(N\) is the number of ray segments, \(T_{p0}(\theta_{10}^{n})\) is the pressure transmission coefficient at the \(n\)-th incidence angle, \(\bar{r}_{10}^{n}\) and \(\bar{r}_{20}^{n}\) are the normalized path lengths in fluid and solid media, \(\theta_{10}^{n}\) and \(\theta_{20}^{n}\) are the incidence and refraction angles, and \(D_{\frac{b}{N}}(\theta)\) is the directivity function of each sub-element. Basically, it captures the amplitude, phase, and directional effects induced by the interface and is also implemented in Python.

\subsection{Point Source Model}

In contact or immersion ultrasonic inspections involving fluid-solid interfaces, the velocity field radiated by each subelement of a rectangular array can be modeled as originating from a 3-D point source, according to high-frequency ray theory~\cite{b1}. This formulation accounts for the specific propagation modes in each medium, where \(c_{p1}\) represents the compressional wave speed in the fluid and \(c_{\beta2}\) denotes the transmission wave speed in the solid, which may be compressional (\(\beta = p\)) or shear (\(\beta = s\)).

For a rectangular transducer element subdivided into \(R \times Q\) subelements, the velocity field \(\mathbf{v}(\mathbf{x}, \omega)\) at an arbitrary field point \(\mathbf{x}\), for a given angular frequency \(\omega\), is calculated by summing the contributions of each subelement along its ray path through both media. The detailed expression is given in eq.~\eqref{eq:ps3Dint_contact}, adapted from the canonical beam models in Schmerr~\cite[Sec.~6.5]{b1}.

\begin{align}
\mathbf{v}(\mathbf{x}, \omega) &= \frac{-i k_{p1} p_0(\omega) \, \Delta d_x \Delta d_y}{2 \pi \rho_1 c_{p1}} \notag \\
&\quad \times 
\sum_{r=1}^{R} \sum_{q=1}^{Q} K_p(\bar{\theta}^{rq}) \, \bar{T}_{ss}^{\beta;p}(\bar{\theta}_1^{rq}) \, \bar{\mathbf{d}}_{\beta2}^{rq} \notag \\
&\quad \times \frac{\sin\left(k_{p1} u_{x'}^{rq} \Delta d_x / 2\right)}{k_{p1} u_{x'}^{rq} \Delta d_x / 2} 
\cdot \frac{\sin\left(k_{p1} u_{y'}^{rq} \Delta d_y / 2\right)}{k_{p1} u_{y'}^{rq} \Delta d_y / 2} \notag \\
&\quad \times \frac{
\exp\left[i k_{p1} r_1^{rq} + i k_{\beta2} r_2^{\beta;rq} \right]
}{
\sqrt{ \left( r_1^{rq} + \frac{c_{\beta2}}{c_{p1}} \frac{\cos^2(\bar{\theta}_1^{rq})}{\cos^2(\bar{\theta}_2^{rq})} r_2^{\beta;rq} \right)
\left( r_1^{rq} + \frac{c_{\beta2}}{c_{p1}} r_2^{\beta;rq} \right)
}
},
\label{eq:ps3Dint_contact}
\end{align}
where the variables in eq.~\eqref{eq:ps3Dint_contact} are defined as follows: The field point is denoted by \(\mathbf{x}\), and \(\omega\) is the angular frequency. The term \(p_0(\omega)\) is the input pressure spectrum, and \(\Delta d_x\), \(\Delta d_y\) are the dimensions of the elements. The fluid properties are \(\rho_1\) (density) and \(c_{p1}\) (compressional wave speed), and the solid properties include \(c_{\beta2}\), which represents compressional or shear wave speed in the transmitted medium. The wavenumbers are defined as \(k_{p1} = \omega / c_{p1}\) and \(k_{\beta2} = \omega / c_{\beta2}\). The term \(K_p(\bar{\theta}^{rq})\) denotes the geometric scaling factor, and \(\bar{T}_{ss}^{\beta;p}(\bar{\theta}_1^{rq})\) is the mode-converted transmission coefficient. The unit polarization vector is \(\bar{\mathbf{d}}_{\beta2}^{rq}\), while \(u_{x'}^{rq}\), \(u_{y'}^{rq}\) represent the directional cosines in local coordinates. The lengths of the ray paths are given by \(r_1^{rq}\) (in the fluid) and \(r_2^{\beta;rq}\) (in the solid), and the angles \(\bar{\theta}_1^{rq}\), \(\bar{\theta}_2^{rq}\) are the incidence and refraction angles, respectively. 
These quantities are evaluated for each subelement \((r,q)\) of the discretized rectangular source model.

\subsection{Delay Law Formulation}

To accurately steer and focus ultrasonic beams through a planar interface between two different media, such as water and steel, time delay laws must be formulated to correct for the refraction-induced path differences among transducer elements.

The general delay time \(\Delta t_{mn}^{d}\) for the \( (m,n) \)-th element in a 2-D array is computed differently depending on whether the goal is steering alone or both steering and focusing. These are formalized as follows, based on the derivations in Schmerr \cite{b1}:

For steering alone:

\begin{equation}
\Delta t_{mn}^{d} = \left| \left\{ \Delta t_{mn} \right\}_{\text{min}} \right| + \Delta t_{mn},
\label{eq:delay_law_steering}
\end{equation}

and for steering with focusing through a planar interface:

\begin{equation}
\Delta t_{mn}^{d} = \left( \Delta t_{mn} \right)_{\text{max}} - \Delta t_{mn},
\label{eq:delay_law_focusing}
\end{equation}
where \( \Delta t_{mn} \) is the time of flight from element \((m,n)\) to the focal point, \( \Delta t_{mn}^{d} \) is the final digital delay applied to element \((m,n)\), and \((\Delta t_{mn})_{\text{min}}, (\Delta t_{mn})_{\text{max}} \) are the minimum and maximum geometric delays across the array. The ray paths are computed considering fluid and solid wave velocities \( c_1 \) and \( c_2 \), a planar interface at depth \( DT_0 \), and focal depth \( DF \).

These expressions ensure that all delays are non-negative and aligned for digital implementation. The term \(\Delta t_{mn}\) represents the geometric delay associated with the travel time from the \( (m,n) \)-th element to the focal point \( \mathbf{x} \), accounting for the refraction at the interface and the different sound speeds in the two media. A program implements both equations using geometric ray tracing based on Snell’s law. It supports arbitrary steering angles \(\theta_T\), \(\phi\), a refraction angle \(\theta_2\), array geometry, and interface geometry \cite{b1}.

\subsection{Quantitative Evaluation}

To quantitatively evaluate the performance of the simulation pipeline and the effectiveness of the synthetic focusing, we compute the FWHM of the beam envelope along defined cross sections of interest. It is widely recognized as a robust metric for characterizing resolution in imaging systems, as it measures the width of a response function at 50\% of its peak amplitude. In this study, we apply a definition-based method where the peak amplitude is located, and the width of the beam profile at 50\% of this maximum is estimated by interpolation.

\begin{equation}
\text{FWHM} = x_2 - x_1, \quad \text{where} \quad A(x_1) = A(x_2) = 0.5 \cdot A_{\text{max}},
\label{eq:fwhm}
\end{equation}
where \(x_1\) and \(x_2\) are the positions along the beam envelope where the amplitude \(A(x)\) drops to 50\% of its maximum value. These points correspond to the left and right crossings of the beam profile at half its peak, and the FWHM is defined as the distance between them \cite{b15}. To increase precision, the crossing points at half the maximum amplitude are estimated using local quadratic interpolation around the envelope samples.

Several estimation methods for FWHM exist in the literature, including approaches based on Gaussian fitting, statistical moments, and standard deviation scaling. A recent comparative analysis of seven methods by Rainio \textit{et al.}~\cite{b7} shows that definition-based and NEMA-standard methods provide reliable results, particularly when applied to sparse or noisy datasets. These findings reinforce our choice of using a direct definition-based interpolation technique, which ensures robustness and interpretability in the simulation results.

In addition to the main definition-based measurement, we evaluated multiple FWHM estimation strategies as proposed by Rainio \textit{et al.}~\cite{b7}, labeled F1 through F7. These include: F1 - definition-based estimation (peak to half-max), F2 - area-based estimation from envelope energy, F3 - moment-based estimation using statistical variance, F4 - log-parabola fit to the envelope, F5 - slope-based width from a logarithmic fit, F6 - local quadratic interpolation around half-max crossings, and F7 - Gaussian model fitting via optimization. Each method was compared with a theoretical FWHM baseline derived from classical diffraction theory.

\begin{equation}
\text{FWHM}_{\text{theory}} = \frac{c_2 \cdot 1000}{f \cdot 10^6} \cdot \text{DF},
\label{eq:fwhm_theory}
\end{equation}
where \(c_2\) is the longitudinal velocity in the solid, \(f\) is the central frequency (MHz), and DF is the F-number. This multimethod approach provides robustness and confidence in our quantitative beam evaluation. For smoothing of the TFM envelope, a Gaussian filter with standard deviation ($\sigma$) from 0.49 to 5.06 was used, balancing noise reduction and preservation of the peak shape.

\subsection{Conceptual Foundations of the Digital Twin-Driven Paradigm}

The term digital twin-driven refers to a paradigm in which a DT is not merely a passive representation of a physical system but actively influences and guides decision-making processes throughout the system lifecycle. In this approach, the DT becomes a core enabler of intelligent behavior in design, operation, monitoring, and optimization tasks.

Tao \textit{et al.}~\cite{b20} introduced the concept of digital twin-driven manufacturing as a strategy in which physical product data, virtual product data, and the connected data that connect them are leveraged in real time to improve efficiency, intelligence, and sustainability in product design, production, and service. This driving role arises from the seamless convergence of cyberphysical spaces, allowing simulations to continuously reflect operational states, and thus direct lifecycle activities.

Expanding on this, Liu \textit{et al.}~\cite{b21} applied digital twin-driven thinking to enable the rapid individualized design of automated flow-shop manufacturing systems, using DT to drive layout generation and system configuration in real-time. Leng \textit{et al.}~\cite{b22} further demonstrated the potential for rapid digital twin-driven reconfiguration of manufacturing systems, emphasizing the active role of DT in supporting system responsiveness through open architecture models.

These applications underscore a shift from using DTs as static digital models to employing them as real-time, dynamic agents for system control. This distinction is reinforced by recent reviews. For example, Attaran and Celik~\cite{b23} emphasized that in digital twin-driven systems, DT is not only descriptive but also predictive and prescriptive, orchestrating decision making through continuous feedback and analytics. Similarly, Semeraro \textit{et al.}~\cite{b24} noted that such systems transition from reactive to proactive strategies, where DT continuously drives improvement and adaptation.

Finally, Zhang \textit{et al.}~\cite{b25} and Yang \textit{et al.}~\cite{b26} both affirm that digital twin-driven frameworks are characterized by closed-loop feedback, real-time synchronization, and autonomy. In these systems, the DT serves as the operational hub, steering execution based on predictions, simulations, and optimized control policies. In summary, while a digital twin-based simulation primarily utilizes the digital twin as its foundation, a digital twin-driven simulation involves the digital twin actively controlling or guiding the simulation process.

\subsection{Digital Twin-Driven Simulation Methodology for Ultrasonic Phased Arrays}

Digital twin technology has emerged as a transformative tool for NDE, enabling the real-time coupling of physical systems with virtual models to enhance prediction, monitoring, and control capabilities. In ultrasonic phased array inspection, this approach enables virtual replication of signal propagation and response from complex geometries, offering a valuable complement to experimental data. According to Filho and Bélanger~\cite{b16}, the integration of DT with robotic automation and the global total focus method (gTFM) allows full coverage of the components with minimal user intervention, demonstrating the efficacy of DT in guiding inspection procedures and optimizing signal acquisition.


Validation of the DT is achieved by comparing the resulting TFM image fields with expected beam behaviors through quantitative metrics. Lai \textit{et al.}~\cite{b2} present a DT approach that combines measured displacement data with computational predictions using an optimization scheme, improving the precision of the model to monitor the structural health of composite wings. In parallel, the framework presented by Huang \textit{et al.}~\cite{b18} separates the DT process into offline and online phases, where the offline stage constructs the simulation based on design data and the online stage updates it with test results. This separation is essential for NDE, where access to the internal state is limited and evaluation relies heavily on image-based indicators such as lateral resolution and focal width.

The methodology aligns with the general purpose DT architecture proposed by Tan \textit{et al.}~\cite{b19}, in which the physical system is mirrored by a virtual model that continuously ingests system data to support decision making. Their framework, although applied to hydroelectric power operations, highlights essential principles applicable across domains, including model-driven simulation, real-time monitoring, and feedback-based validation. In particular, the incorporation of quantitative evaluation strategies for fault detection reinforces the importance of employing objective metrics in DT validation. Although their metrics relate to classification performance, the underlying principle supports the use of domain-specific indicators, such as FWHM, in imaging-based DT applications. Furthermore, their use of deep learning modules to improve system resilience illustrates how predictive mechanisms can be integrated within a DT to diagnose deviations from expected performance.

Together, the integration of simulation, virtual focus, and quantitative evaluation in a DT framework enhances the methodological robustness of the simulation of the ultrasonic phased array. It enables not only the design and validation of inspection scenarios, but also the creation of synthetic data to inform and refine imaging strategies in complex geometries.

\section{Digital Twin-Driven Simulation Framework}
\label{sec:DT-driven_simulation_framework}

The proposed simulation framework adopts the DT paradigm to replicate the behavior of ultrasonic phased arrays through fluid-solid interfaces. As a virtual mirror of the real inspection system, the framework is capable of modeling pressure and velocity fields, synthesizing delay laws, and reconstructing focused ultrasonic images using FMC-TFM techniques. This virtualized inspection system facilitates parametric exploration and repeatability, while eliminating experimental complexity and cost.

The framework is implemented in Python and is structured into modules, each replicating a distinct part of the acquisition and reconstruction process. It builds on the canonical mathematical models described by Schmerr \cite{b1}, with custom adaptations to handle interface-induced propagation effects.

The simulation pipeline includes the following components:

\begin{itemize}
    \item \textbf{Field Modeling:} The pressure and velocity fields are simulated using physically grounded beam propagation models derived from ray theory and Kirchhoff-type approximations~\cite{b1}. These models incorporate phenomena such as wave speed discontinuities, acoustic transmission coefficients, Snell’s law for refraction, and directivity effects across fluid-solid interfaces.
    
    \item \textbf{Delay Law Computation:} Time delays for beam steering and focus through planar interfaces are computed using custom numerical algorithms developed within this framework~\cite{b1}. These methods consider the geometric paths of rays across material boundaries and apply refraction principles to ensure accurate focusing.

    \item \textbf{Synthetic Signal Generation:} Using a virtual FMC scheme, the framework simulates transmit–receive responses for all pairs of elements, generating input data for the TFM stage. The concept of FMC as a complete transmission-receive acquisition strategy was introduced and widely formalized in \cite{b12}, providing the foundation for post-processed imaging such as TFM. The beam summation incorporates apodization and interpolated delay compensation.

    \item \textbf{Image Reconstruction:} A Python-based TFM routine processes FMC data to synthetically focus at all image points. This mirrors physical systems and enables precise analysis of focus performance and resolution. The TFM image reconstruction algorithm, based on delay-and-sum beamforming with time-of-flight (ToF) compensation, was established by Holmes \textit{et al.}~\cite{b13} and has become a cornerstone of high-resolution ultrasonic imaging.

    \item \textbf{Post-Processing and Evaluation:} A dedicated module evaluates focus quality using FWHM, beam profiles, and alignment with theoretical predictions \cite{b7}.
\end{itemize}

The modular structure supports rapid parameter studies, sensitivity analysis, and system optimization. This approach is aligned with the DT strategies adopted in the recent literature on ultrasonic inspections~\cite{b9, b16}. Specifically, the use of ray-based wave modeling \cite{b10} and the refinement of the delay and sum (DAS) method \cite{b11} support our implementation as a viable DT for phased array inspection scenarios.

\section{FMC-TFM Imaging and Quantification Methodology}
\label{sec:FMC-TFM_imaging_and_quantification_methodology}

% Describe how the FMC matrix is generated and processed
% Explain how TFM reconstruction is performed
% Present how FWHM, envelope detection, and delay law comparison are computed

This section outlines the core simulation workflow used to replicate ultrasonic phased array inspection using the FMC and TFM pipeline.

\subsection{FMC Data Simulation}
A complete synthetic FMC dataset is constructed by simulating the time-domain response for every transmit–receive pair in the array. Each transmitter is excited sequentially, while pressure or velocity responses are computed at all receiver positions using the field models described in Section~\ref{sec:theoretical_background}. For each focal law, the paths are corrected for interface effects (refraction, travel time, and beam spread). The data is stored in a three-dimensional array indexed by transmit element, receive element, and time sample.

\subsection{Delay-and-Sum TFM Reconstruction}
The image reconstruction step uses the TFM as introduced by Holmes \textit{et al.}~\cite{b13}. For each pixel in the image grid, all corresponding transmit-receive paths are evaluated using geometric delay calculations from the \texttt{delay\_laws\_3D\_int} function. The respective time-domain signals are compensated for by delay and summed coherently to synthesize a focused response at each point. Interpolation is applied to align non-integer delays.

\subsection{Envelope Detection and Normalization}
After reconstruction, the A-scan signal envelope is extracted using the analytic signal through the Hilbert transform. Each envelope is normalized relative to the global image maximum to facilitate consistent FWHM measurements. Beam profiles are extracted along the central axis or focal plane depending on the analysis.

\subsection{Evaluation Procedure}
The envelope data are used to compute the FWHM as described in Section~\ref{sec:theoretical_background}. Multiple beam positions and parameter configurations (focal depth, angle, frequency) are tested to quantify system resolution. The simulation results are compared with the theoretical expectations derived from equation~\eqref{eq:fwhm_theory}, and the discrepancies are analyzed in light of interface distortions and delay law interpolation artifacts.

This methodology ensures that the DT framework is not only structurally faithful but also diagnostically relevant, enabling quantitative benchmarking against analytical standards and recent advances in phased array modeling.

\section{Results and Discussion}
\label{sec:results_and_discussion}

The proposed DT framework was evaluated through a series of synthetic experiments, each replicating realistic conditions for ultrasonic phased array simulation across fluid-solid interfaces. This section presents the core results and critical analysis of focus performance, delay law precision, and FWHM estimation across multiple test configurations.

\subsection{Test Configuration and Parameters}

All simulations used an 1D linear array immersed in water, targeting a steel half-space through a planar interface. Each array configuration consisted of 64 elements with an element width of 0.45 mm, kerf (gap) of 0.05 mm, and an elevation aperture of 10 mm. The excitation frequency was set to 5 MHz and the sampling rate was set to 50 MHz. The water and steel properties were fixed throughout all experiments, with sound speeds of 1480 m/s (water), 5900 m/s (steel longitudinal) and 3200 m/s (steel shear).

Table~\ref{tab:sim_config} summarizes key simulation parameters for each test case, including window type, refracted angle, focal design (F-number) and scanning grid.

\begin{table}[htbp]
\caption{Simulation Parameters per Test Case}
\begin{center}
\begin{tabular}{|p{.1cm}|p{1cm}|p{.5cm}|p{.5cm}|p{1.2cm}|p{0.9cm}|p{1cm}|p{.1cm}|}
\hline
\textbf{\rotatebox[origin=c]{90}{Test}} & \textbf{Window Function} & \textbf{DF (mm)} & \textbf{$D_t0$ (mm)} & \textbf{X-Scan Range (mm)} & \textbf{Z-Scan Range (mm)} & \textbf{Grid (X,Z)} & \textbf{\rotatebox[origin=c]{90}{Angle}} \\
\hline
1&rect&25&15&-21.2:22.2&1:89.8&450×900&0° \\
2&Hamm&25&15&-5:5&10:50& 21×81& 0° \\
3&rect&25&15&-20:20&1:101&81×201&0° \\
4&Hamm&25&15&-20:20&1:101&81×201&0° \\
5&rect&25&15&-5:5&10:55&81×201&0° \\
6&Hamm&25&15&-5:5&10:55&81×201&0° \\
7&rect&15&15&-25:25&1:51&51×50&0°\\
\hline
\end{tabular}
\label{tab:sim_config}
\end{center}
\end{table}

The interface depth was fixed at 15 mm across all tests, and the focal depth varied through the F-number and scanning grid definitions. Each simulation calculated the appropriate delay laws through Snell-based tracing and followed a full FMC transmission strategy, storing raw A-scans for all transmit–receive element pairs. Apodization windows were varied to assess their effect on beam focusing and sidelobes suppression.

\subsection{Delay Law Resolution and Sampling Accuracy}
    
In all tests (1-7), the delay values computed in the elements of the transducer array ranged approximately from 0.6875 to 0.9893 samples. These values were obtained by calculating the geometric ToF between each array element and the focal point, and then converting these delay times from microseconds to discrete samples using the system sampling frequency of 50 MHz. Given a sampling period of \( T_s = 1 / f_s = 20\ \text{ns}\), a delay of 0.6875 samples corresponds to \( 0.6875 \times 20 = 13.75\ \text{ns} \), while 0.9893 samples correspond to \( 19.79\ \text{ns} \).
    
In digital beamforming systems, it is widely accepted that delay quantization errors should remain below 0.25 samples (i.e., \(0.25 \times 20 = 5\ \text{ns} \)) to avoid significant degradation of focal quality. This design threshold ensures that time delays can be closely approximated, minimizing interpolation artifacts, and preserving the integrity of the synthesized beam pattern. Previous studies have shown that coarse quantization exceeding this threshold adversely affects the fidelity of the main lobe, raises sidelobe levels, and introduces artifacts from the grating lobe~\cite{b27,b14}. However, in all simulated test cases, this delay resolution tolerance was violated. The sampling check analysis revealed pronounced misalignments between the continuous-time delay requirements and the discrete-time implementation grid, resulting in quantization-induced distortions. These effects are illustrated in fig.~\ref{fig:delay_heatmap}, which presents a delay law heatmap, and fig.~\ref{fig:delay_stem} illustrates the delay law across all elements in test 7.

The axes are labeled using the notation \text{TxN} and \text{RxN} in fig.~\ref{fig:delay_heatmap}, where \text{N} indicates the element index for the transmit and receive channels, respectively. Because the array is configured as a one-dimensional 64×1 linear array, the delay matrix can be interpreted with respect to a single receiver channel, commonly designated as Rx0 for visualization purposes. In the context of FMC, each element of the array transmits sequentially while all others receive, generating a complete dataset of transmission-reception pairs. But the delay visualizations shown here do not represent the raw FMC acquisition timings. Instead, they correspond to the focusing delays used during TFM reconstruction, derived from geometric ToF calculations relative to a focal point. For this reason, the delay heatmap presents a fixed receiver index (Rx0) along the x-axis and spans all transmitters (Tx0 to Tx63) along the y-axis. This format of labels improves clarity by explicitly distinguishing the functional role and index of each element, even when only one axis is varied for simplicity. The horizontal dashed line in fig.~\ref{fig:delay_stem} marks the 0.25-sample threshold, confirming that most delays exceed the safe quantization limit, resulting in distortion and under-sampling. This supports the observed performance degradation in beam focus and FWHM fidelity for the affected cases.    
    
    \begin{figure}[htbp]
    \centerline{\includegraphics[width=\linewidth]{test0007_delay_laws_heatmap.png}}
    \caption{Delay law heatmap for test 7, showing element-wise delay values used in the FMC acquisition. The distribution reflects nonlinear propagation paths due to refraction at the fluid-solid interface, resulting in delay values ranging approximately from 0 to 0.9893 samples. The observed curvature of the delay map highlights focusing through the steel interface. Axes are labeled as \text{TxN} and \text{RxN}, where \text{N} denotes the element index for transmit and receive elements, respectively, to enhance interpretability of the array layout.}
    \label{fig:delay_heatmap}
    \end{figure}
    
    \begin{figure}[htbp]
    \centerline{\includegraphics[width=\linewidth]{test0007_delay_laws_stem.png}}
    \caption{The stem plot for the central element (element 32) in test 7 illustrates a parabolic delay profile degraded by staircase-like sampling, indicative of insufficient temporal resolution. Delays exceeding the 0.25-sample threshold confirm quantization errors, impacting focal quality and necessitating interpolation.}
    \label{fig:delay_stem}
    \end{figure}
    

\subsection{TFM Envelope and Focusing Quality}

The reconstruction of the TFM envelope, fig.~\ref{fig:tfm_envelope} shows the spatial concentration of acoustic energy near the focal point. Among all simulations, only test 2 demonstrated accurate focusing, with an error of FWHM below 5\%, confirming good agreement between synthetic focusing and theoretical expectations. In contrast, tests 1, 3, 5 and 7 exhibited significant beam spreading and degradation, with FWHM errors ranging from 44\% to more than 165\%, indicating reduced focus quality due to delay misalignment or envelope resolution limitations, fig.~\ref{fig:tfm_envelope_poor}.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\linewidth]{test0002_tfm_envelope.png}}
\caption{TFM envelope image for test 2 using a Hamming apodization. The beam energy is well-focused at the designated depth, producing a sharp, symmetric main lobe. The measured FWHM (30.95 mm) closely matches the theoretical expectation (29.50 mm), with an error below 5\%. The interface effect is minimized due to optimal delay quantization and apodization smoothing, preserving beam coherence.}
\label{fig:tfm_envelope}
\end{figure}

\begin{figure}[htbp]
\centerline{\includegraphics[width=\linewidth]{test0007_tfm_envelope.png}}
\caption{For test 7, the TFM envelope shows severe degradation with lateral spreading and elevated sidelobe energy, indicating defocusing. Refraction-induced distortion from the steel interface, compounded by coarse delay sampling, caused this degradation. The F1 FWHM (47.33 mm) deviates over 165\% from the theoretical value (17.70 mm), highlighting the detrimental effect of quantization errors and the need for subsample delay compensation.}
\label{fig:tfm_envelope_poor}
\end{figure}

\subsection{FWHM Evaluation across Methods}

Table~\ref{tab:fwhm_summary} summarizes the FWHM results of the definition-based method (F1) and the alternative estimators F2 to F7. Methods F4 and F5 consistently yielded values close to theoretical expectations, while F6 and F7 failed in most tests due to numerical instability or poor model fit. The definition-based approach (F1) overestimated FWHM in nearly all cases due to coarse envelope resolution and delay misalignment. 

\begin{table}[htbp]
\caption{Summary of FWHM Estimations (Tests 1–7)}
\begin{center}
\scalebox{0.85}{
\begin{tabular}{|p{.3cm}|p{.8cm}|p{.8cm}|p{.4cm}|p{.45cm}|p{.4cm}|p{.4cm}|p{.45cm}|p{.45cm}|p{.35cm}|p{.35cm}|p{.35cm}|}
\hline
\textbf{Test} &\textbf{F1(mm) \rotatebox{90}{(Theoretical)}}& \textbf{F1(mm) \rotatebox{90}{(Measured)}} & \textbf{$\sigma$} & \textbf{F1 (\%)} & \textbf{F2 (\%)} &  \textbf{F3 (\%)} &  \textbf{F4 (\%)} &  \textbf{F5 (\%)} &  \textbf{F6 (\%)} &  \textbf{F7 (\%)}\\
\hline
1 &29.50& 63.09 & 5.06 & 113.9 & 34.6 & 17.6 & 17.3 & 20.0 & 90.4 & 85.7 \\
2 &29.50& 30.95 & 0.60 & 4.9 & 36.9 & 35.1 & 15.3 & 19.9 & NaN & 84.8 \\
3 &29.50& 51.53 & 1.99 & 74.7 & 14.4 & 15.9 & 14.2 & 15.4 & NaN & 84.7 \\
4 &29.50& 51.53 & 1.99 & 74.7 & 14.4 & 15.9 & 14.2 & 15.4 & NaN & 84.7 \\
5 &29.50& 42.47 & 1.33 & 44.0 & 40.5 & 34.6 & 12.6 & 16.3 & NaN & 86.0 \\
6 &29.50& 42.47 & 1.33 & 44.0 & 40.5 & 34.6 & 12.6 & 16.3 & NaN & 86.0 \\
7 &17.70& 47.33 & 0.49 & 167.4 & 15.8 & 10.6 & 372.0 & 335.8 & NaN & 40.4 \\
\hline
\end{tabular}}
\label{tab:fwhm_summary}
\end{center}
\end{table}

To highlight the consistency of the estimation, fig.~\ref{fig:fwhm_methods_test02} shows the seven FWHM estimation outputs for the well-focused case of test 2.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\linewidth]{test0002_Comparison of FWHM Estimation methods (F1-F7).png}}
\caption{Comparison of seven FWHM estimation methods (F1-F7) for test 2. The blue dashed line represents the theoretical value (29.50 mm). Methods F4 and F5 show close agreement, while F6 and F7 yield extreme deviations due to poor numerical stability.}
\label{fig:fwhm_methods_test02}
\end{figure}

\subsection{Scientific Implications}

The discrepancy between analytical and simulated FWHM highlights practical challenges in phased array implementation, especially in the presence of refraction and delay quantization. Our results quantify resolution degradation and emphasize the importance of subsample delay interpolation and signal conditioning.

Moreover, by benchmarking seven FWHM methods under controlled simulation conditions, we contribute a reproducible framework for evaluating focusing performance, which could inform delay law design, hardware constraints, and system calibration.

Finally, fig.~\ref{fig:digital_twin_field} illustrates the simulated acoustic pressure field generated by the digital twin for the off-axis focal configuration in test 1.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\linewidth]{test0001_digital_twin_field.png}}
\caption{Acoustic pressure field generated by the digital twin for test 1. The left panel shows the full beam profile across the water-steel interface, highlighting beam refraction at the planar boundary near \( z = 15\ \text{mm} \), which introduces curvature and asymmetry in the focal path. The right panel presents an adaptive zoom of the \(-20\ \text{dB}\) beam region, restricted to the effective focal area with detectable amplitude.}
\label{fig:digital_twin_field}
\end{figure}

\subsection*{Impact of the Fluid-Solid Interface}

The fluid-solid interface introduces wave refraction governed by Snell’s law, resulting in non-linear wavefront propagation and altered ToF trajectories for each transducer element. These distortions become especially critical in TFM reconstructions, where precise geometric delay laws are required to ensure constructive interference at the focal point.

As shown in fig.~\ref{fig:digital_twin_field}, the beam undergoes visible bending and spreading as it traverses the interface, leading to partial distortion of the wavefront. This behavior is consistent with the observations reported by Mineo et al.~\cite{b28}, who demonstrate that the presence of even a single interface significantly increases the complexity of ray paths and introduces focal drift and refraction-induced delays that deviate from those predicted under homogeneous assumptions. Their study confirms that improperly modeled interfaces, whether planar or curved, can introduce systematic errors in delay law calculations, ultimately leading to degraded beam quality.

Moreover, Mineo et al.~\cite{b28} emphasize that conventional ray tracing methods are insufficient when interfaces are present, and they propose an iterative optimization scheme to calculate accurate incidence angles and delay profiles across multilayered structures. The authors show that in the absence of such corrections, ultrasonic beams exhibit significant misalignment, directly affecting focusing accuracy and spatial resolution.

In our results, this theoretical insight is corroborated by the quantitative degradation observed in tests 1, 3, 4, and 7, where the interface-induced refraction, compounded by coarse delay sampling, results in substantial FWHM overestimations and focal distortion. These findings reinforce that the interface is not a secondary effect but a dominant factor in TFM resolution loss, specifically in scenarios lacking delay law refinement or subsample interpolation.

Therefore, consistent with the conclusions of Mineo et al.~\cite{b28}, the presence of the fluid-solid interface in our digital twin simulations plays a central role in the observed degradation of FWHM and beam profile for the most affected test cases.

\section{Conclusion}
\label{sec:conclusion}

This work presented a DT-driven simulation framework for modeling and analyzing the inspection of ultrasonic phased arrays through fluid-solid interfaces. By combining canonical beam models, Snell-based delay law computation, and a synthetic FMC-TFM pipeline, the proposed system enables reproducible and physically grounded simulation of focusing performance and field propagation.

Quantitative evaluation based on FWHM analysis in seven methods revealed a significant variation in beam resolution depending on delay quantization, apodization settings, and standard deviation ($\sigma$) used for envelope smoothing. The framework successfully replicated key characteristics of ultrasonic focusing. The results confirmed that model-based estimators (F4, F5) consistently offer strong alignment with theoretical predictions in various test configurations (e.g., tests 1, 3-6 in Table~\ref{tab:fwhm_summary}). Although the definition-based method (F1) demonstrated high accuracy in specific cases (e.g., test 2 with 4.9\% error), it often overestimated FWHM in others due to the coarseness or distortion of the envelope, especially with higher $\sigma$ values (e.g., test 7 with $\sigma=5.06$, yielding 113.9\% error). Furthermore, methods F6 and F7 frequently exhibited numerical instability or poor model fit.

Then, the results demonstrate that the presence of a fluid-solid interface significantly contributes to the degradation of beam quality, particularly when compounded by coarse delay sampling and interpolation limitations. This interaction, thoroughly analyzed in Section~\ref{sec:results_and_discussion}, underscores the necessity of high-resolution delay control when modeling across acoustic impedance mismatches. These findings demonstrate the potential of the framework as a reliable testbed for phased array simulation development, parameter sensitivity studies, and signal processing benchmarking. Future work will focus on experimental validation, extension to anisotropic and multilayered media, and AI-assisted optimization of inspection configurations, all grounded in the DT framework.

\section*{Acknowledgment}
\label{sec:acknowledgment}

The support of the "Laboratório de Ultrassom" (LUS) and infrastructure from the University of São Paulo are gratefully acknowledged.

\begin{thebibliography}{00}
\bibitem{b3} C. Anand, R. M. Groves \& R. Benedictus, "Modeling and Imaging of Ultrasonic Array Inspection of Side Drilled Holes in Layered Anisotropic Media," {\em Sensors}. \textbf{21} (2021), doi: 10.3390/s21144640.
\bibitem{b6} C. Holmes, B. W. Drinkwater \& P. D. Wilcox, "Post-processing of the full matrix of ultrasonic transmit–receive array data for non-destructive evaluation," {\em NDT \& E International}, \textbf{38}, 701-711 (2005), doi: 10.1016/j.ndteint.2005.04.002.
\bibitem{b2} X. Lai, L. Yang, X. He, Y. Pang, X. Song, and W. Sun, "Digital twin-based structural health monitoring by combining measurement and computational data: An aircraft wing example," Journal of Manufacturing Systems, vol. 69, pp. 76-90, Jun. 2023, doi: 10.1016/j.jmsy.2023.06.006.
\bibitem{b1} L. W. Schmerr Jr., "Fundamentals of Ultrasonic Phased Arrays, 1st ed., Solid Mechanics and Its Applications," vol. 215, Cham: Springer, 2015, doi: 10.1007/978-3-319-07272-2.
\bibitem{b15} A. Khorshidi \& M. Ashoor, "Quantitative assessment of full-width at half-maximum and detector energy threshold in X-ray imaging systems," {\em European Journal Of Radiology}, \textbf{176} pp. 111537 (2024), doi: 10.1016/j.ejrad.2024.111537.
\bibitem{b7} O. Rainio, J. Hällilä, J. Teuho \& R. Klén, "Methods for estimating full width at half maximum," {\em Signal, Image And Video Processing}, \textbf{19}, 289 (2025), doi: 10.1007/s11760-025-03820-6.
\bibitem{b20} F. Tao, J. Cheng, Q. Qi, M. Zhang, H. Zhang \& F. Sui, "Digital twin-driven product design, manufacturing and service with big data," The International Journal of Advanced Manufacturing Technology, vol. 94, no. 9, pp. 3563-3576, Feb. 2018, doi: 10.1007/s00170-017-0233-1.
\bibitem{b21} Q. Liu, H. Zhang, J. Leng, \& X. Chen, "Digital twin-driven rapid individualised designing of automated flow-shop manufacturing system," International Journal of Production Research, vol. 57, no. 12, pp. 3903-3919, 2019, doi: 10.1080/00207543.2018.1471243.
\bibitem{b22} J. Leng, Q. Liu, S. Ye, J. Jing, Y. Wang, C. Zhang, D. Zhang, \& X. Chen, "Digital twin-driven rapid reconfiguration of the automated manufacturing system via an open architecture model," Robotics and Computer-Integrated Manufacturing, vol. 63, Art. no. 101895, 2020, doi: 10.1016/j.rcim.2019.101895.
\bibitem{b23} M. Attaran \& B. G. Celik, "Digital Twin: Benefits, use cases, challenges, and opportunities, "Decision Analytics Journal, vol. 6, Art. no. 100165, 2023, doi: 10.1016/j.dajour.2023.100165.
\bibitem{b24} C. Semeraro, M. Lezoche, H. Panetto \& M. Dassisti, "Digital twin paradigm: A systematic literature review," Computers in Industry, vol. 130, Art. no. 103469, 2021, doi: 10.1016/j.compind.2021.103469.
\bibitem{b25} Z. Bing, M. Enyan, J. N. O. Amu-Darko, E. Issaka, L. Hongyu, R, Junsen, Z. Xinxing, "Digital twin on concepts, enabling technologies, and applications," Journal of the Brazilian Society of Mechanical Sciences and Engineering, vol. 46, no. 7, Art. no. 420, Jul. 2024, doi: 10.1007/s40430-024-04973-0.
\bibitem{b26} C. Yang, H. Yu, Y. Zheng, L. Feng, R. Ala-Laurinaho, \& K. Tammi, "A digital twin-driven industrial context-aware system: A case study of overhead crane operation," Journal of Manufacturing Systems, vol. 78, pp. 394-409, 2025, doi: 10.1016/j.jmsy.2024.12.006.
\bibitem{b16} J. F. M. R. Filho \& P. B\'{e}langer, "Global total focusing method through digital twin and robotic automation for ultrasonic phased array inspection of complex components," NDT \& E International, vol. 137, Art. no. 102833, July 2023, doi: 10.1016/j.ndteint.2023.102833.
\bibitem{b18} L. Huang, Z. Xu, T. Gao, X. Liu, Q. Bi, B. Wang \& K. Tian, "Digital twin-based non-destructive testing method for ultimate load-carrying capacity prediction," Thin-Walled Structures, vol. 204, Art. no. 112223, Nov. 2024, doi: 10.1016/j.tws.2024.112223.
\bibitem{b19} J. Tan, R. M. Radhi, K. Shirini, S. S. Gharehveran, Z. Parisooz, M. Khosravi \& H. Azarinfar, "Innovative framework for fault detection and system resilience in hydropower operations using digital twins and deep learning," Scientific Reports, vol. 15, Art. no. 15669, May 2025, doi: 10.1038/s41598-025-98235-1.
\bibitem{b12} C. Holmes, B. W. Drinkwater \& P. D. Wilcox, "Advanced post-processing for scanned ultrasonic arrays: Application to defect detection and classification in non-destructive evaluation," {\em Ultrasonics}, \textbf{48}, 636-642 (2008), Selected Papers from ICU 2007l, doi: 10.1016/j.ultras.2008.07.019.
\bibitem{b13} C. Holmes, B. W. Drinkwater \& P. D. Wilcox, "The post-processing of ultrasonic array data using the total focusing method, Insight - Non-Destructive Testing and Condition Monitoring," vol. 46, no. 11, pp. 677-680, Nov. 2004, doi: 10.1784/insi.46.11.677.52285.
\bibitem{b9} Y. Yang, P. Zhang, M. Hao \& S. Wang, "Development and Testing of a Digital Twin Platform for Robotic Ultrasound," {\em 2024 IEEE 4th International Conference On Digital Twins And Parallel Intelligence (DTPI)}, pp. 500-504 (2024), doi: 10.1109/DTPI61353.2024.10778903.
\bibitem{b10} Y. Ohara, M. C. Remillieux, T. J. Ulrich, S. Ozawa, K. Tsunoda, T. Tsuji, T. Mihara, "Exploring 3D elastic-wave scattering at interfaces using high-resolution phased-array system," {\em Scientific Reports}, \textbf{12}, 8291 (2022), doi: 10.1038/s41598-022-12104-9.
\bibitem{b11} M. S. A. Kamarulzaman, R. Hamabe, R. Ohara, S. Sato, S. Izumi \& and H. Kawaguchi, "Ultrasonic Raytracing Simulation Method for Data Augmentation to Surveil the Bathroom with Digital Twins," {\em 2024 IEEE UFFC Latin America Ultrasonics Symposium (LAUS)}, pp. 1-4 (2024), doi: 10.1109/LAUS60931.2024.10552962.
\bibitem{b27} S. Bae, H. Song \& TK. Song, "Analysis of the Time and Phase Delay Resolutions in Ultrasound Baseband I/Q Beamformers," {\em IEEE Transactions On Biomedical Engineering}, \textbf{68}, 1690-1701 (2021), doi: 10.1109/TBME.2020.3019799.
\bibitem{b14} T. Salim, J. C. Devlin and J. Whittington, "Quantization Analysis for Reconfigurable Phased Array Beamforming," (2006), https://api.semanticscholar.org/CorpusID:14443679.
\bibitem{b28} C. Mineo, D. Lines \& D. Cerniglia, "Generalised bisection method for optimum ultrasonic ray tracing and focusing in multi-layered structures," {\em Ultrasonics}, \textbf{111}, 106330 (2021), doi: 10.1016/j.ultras.2020.106330.
\end{thebibliography}
\end{document}

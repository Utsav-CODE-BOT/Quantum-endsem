# Research Limitations, Scientific Discussion & Hardware Constraints

## 1. Classical Simulation Constraints

### 1.1 Exponential Statevector Scaling
Simulating quantum states on classical von Neumann hardware requires tracking statevectors in Hilbert space $\mathcal{H} = \mathbb{C}^{2^n}$:
- For $n = 4$ qubits: Statevector dimension is $2^4 = 16$ complex amplitudes ($\approx 256$ bytes).
- For $n = 8$ qubits: Statevector dimension is $2^8 = 256$ complex amplitudes ($\approx 4$ KB).
- For $n = 16$ qubits: Statevector dimension is $2^{16} = 65,536$ complex amplitudes ($\approx 1$ MB).
- For $n = 30$ qubits: Statevector dimension is $2^{30} \approx 10^9$ complex amplitudes ($\approx 16$ GB RAM).

While 4 to 8 qubits execute efficiently on standard multi-core workstations, evolutionary architecture search evaluates dozens of candidate circuits across multiple generations. Each candidate requires 25 to 50 classical optimizer iterations over 200 data points. Consequently, full simulation runtime scales as:

$$T_{\text{search}} \approx P \times G \times I \times N \times T_{\text{forward}}$$

where $P$ is population size, $G$ is generations, $I$ is optimizer iterations, $N$ is sample size, and $T_{\text{forward}}$ is simulator latency.

---

## 2. Trainability & The Barren Plateau Phenomenon

Variational Quantum Algorithms are fundamentally constrained by the **barren plateau phenomenon** (McClean et al., 2018; Cerezo et al., 2021):
- For randomly initialized, deep, highly entangled circuits, the variance of cost function gradients vanishes exponentially with qubit count:
  $$\text{Var}_{\boldsymbol{\theta}} \left[ \frac{\partial \mathcal{L}}{\partial \theta_k} \right] \in \mathcal{O}\left(\frac{1}{2^n}\right)$$
- In our empirical benchmarks, the Fixed Deep VQC (depth 27, 40 parameters) exhibited degraded test accuracy (92.00%) compared to the shallow (99.00%) and optimized (96.00%) architectures. This empirically confirms that adding variational layers does not monotonically improve classification capacity, and often flattens the optimization landscape.

---

## 3. Quantum Noise Modeling vs Physical Hardware

Our noise robustness benchmarks employ Qiskit Aer's `density_matrix` backend with single-qubit depolarizing, two-qubit depolarizing, and readout bit-flip error channels:
- **Phenomenological Approximations:** Depolarizing noise models isotropic state contraction towards the maximally mixed state $\frac{1}{2^n} \mathbb{I}$.
- **Unmodeled Physical Realities:** Physical superconducting transmons and trapped-ion processors exhibit more complex error physics:
  1. $T_1$ longitudinal energy relaxation and $T_2$ transverse dephasing.
  2. Coherent unitary systematic over/under-rotation errors.
  3. Non-local spectator qubit crosstalk during microwave pulses.
  4. State leakage out of the computational subspace $|0\rangle, |1\rangle$ into higher transmon states $|2\rangle$.
  5. Non-Markovian environmental memory effects.

Future physical deployments should validate the search-optimized architectures on cloud-accessible quantum hardware (e.g., IBM Quantum Eagle/Heron processors).

---

## 4. Scientific Integrity & Absence of "Quantum Advantage" Claims

This project makes **no claim of quantum advantage** over classical machine learning:
1. **Classical Superiority on Benchmark Task:** On the 4-dimensional PCA Fashion-MNIST subproblem, Classical Linear and RBF SVMs achieve 100.00% test accuracy within 0.01 to 0.03 seconds. In contrast, simulated quantum classifiers require 15 to 30 seconds of training time.
2. **Research Contribution:** The genuine research contribution of this work is **automated resource-aware quantum software engineering**:
   - Manually designing quantum ansatz circuits is inefficient and error-prone.
   - This framework demonstrates that an automated multi-objective evolutionary search engine can autonomously discover compact circuit topologies that cut quantum depth by **58.3%** and parameter count by **33.3%** without sacrificing classification accuracy, making them substantially more viable for deployment on error-prone NISQ devices.

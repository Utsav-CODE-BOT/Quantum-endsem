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

## 2. Trainability & Circuit Depth Trade-offs

Variational Quantum Algorithms are often discussed in the context of the **barren plateau phenomenon** (McClean et al., 2018; Cerezo et al., 2021):
- For randomly initialized, deep, highly entangled circuits, theoretical gradient variance vanishes exponentially with qubit count:
  $$\text{Var}_{\boldsymbol{\theta}} \left[ \frac{\partial \mathcal{L}}{\partial \theta_k} \right] \in \mathcal{O}\left(\frac{1}{2^n}\right)$$
- In our empirical benchmarks, the Fixed Deep VQC (depth 27, 40 parameters) exhibited lower test accuracy (92.00%) and required longer training time (51.74s) compared to the shallow (99.00%) and search-optimized (90.00%) architectures. The deeper VQC exhibited lower classification accuracy than the shallow and search-optimized architectures in this experiment. This may reflect optimization difficulties or overparameterization. However, gradient variance was not measured, so the presence of barren plateaus cannot be established from these results. To claim barren plateau behavior rigorously, one would need gradient variance measurements across circuit depths, initializations, and varying system sizes $n$.

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

---

## 5. Dataset Subsetting & Pilot Study Scope (360 Samples)

The empirical validation was conducted on a controlled 360-sample partition:
- **Partition:** 200 training samples, 60 validation samples, and 100 test samples.
- **Task Scope:** Binary classification between Class 0 (T-shirt/top) and Class 9 (Ankle boot), compressed from 784 pixels to 4 orthogonal PCA components.
- **Interpretation Boundary:** With 100 test samples, each single classification outcome alters test accuracy by exactly 1.0 percentage point. The 100.00% test accuracy of classical SVMs is an artifact of this linearly separable 4D binary projection on this sample size. It must not be extrapolated as 100% accuracy on the full 70,000-image, 10-class Fashion-MNIST benchmark. The findings are presented as a controlled pilot experiment.
- **Simulated NISQ Modeling:** All noise experiments represent classical simulations of hardware-inspired noise models via Qiskit Aer's density matrix simulator, not physical quantum device executions.

---

## 6. Analysis of Low-Entanglement Architectures

The evolutionary optimizer converged on a compact architecture (`arch_4169ee9360` with 1 CX gate):
- **Empirical Grounding of Expressive Sufficiency:** While PCA produces orthogonal component directions, this mathematical property does not, by itself, theoretically prove that a non-entangling or low-entangling quantum circuit has sufficient expressive capacity for classification. Rather, our justification is empirical, grounded in the measured performance of the discovered architecture: on this 360-sample 4-feature dataset, the compact circuit with minimal entangling operations achieved 90.00% test accuracy (88.33% validation accuracy) while dramatically cutting exposure to two-qubit gate noise.
- **Multi-Objective Trade-off:** The fitness function explicitly penalizes two-qubit gate count ($N_{2Q}/30$) and noise degradation ($f_{\text{rob}}$). Introducing unnecessary CX gates increases depth and exposes the circuit to two-qubit gate noise without providing marginal classification gain on this specific task.
- **Interpretation Scope:** The optimizer correctly behaved in a resource-aware manner by eliminating unneeded entangling overhead on this specific binary problem. Crucially, this task-specific result is an empirical observation and does not establish that entanglement is unnecessary in quantum machine learning generally or on more complex, higher-dimensional datasets.

---

## 7. Optimization Budget Lineage & Multi-Objective Scalarization

1. **Multi-Objective Scalarization vs Explicit Pareto Frontiers:**
   The optimization algorithm optimizes a scalarized weighted fitness function $F = 0.50 f_{\text{perf}} - 0.30 f_{\text{comp}} + 0.20 f_{\text{rob}}$. It does not run a non-dominated sorting multi-objective genetic algorithm (such as NSGA-II) to compute an exhaustive Pareto front of non-dominated solutions. Visualizations of depth vs. accuracy depict the empirical trade-off among evaluated candidates rather than a formal Pareto front.

2. **Optimizer Iteration Budget Lineage:**
   - In evolutionary architecture search, evaluating each candidate circuit in a population requires training parameterized gates. To keep total search runtime tractable, a budget of `maxiter=25` COBYLA iterations was allocated per candidate (`EXP-SEARCH-4169EE9360`, yielding 90.00% validation and 88.00% test accuracy for `arch_4169ee9360`).
   - In the standardized head-to-head benchmark comparison (`EXP-BENCH-20261120`), all classical and quantum baseline models were evaluated under a uniform protocol of `maxiter=35` iterations, where `arch_4169ee9360` achieved 88.33% validation and 90.00% test accuracy, and the Fixed Standard Baseline achieved 98.33% validation and 95.00% test accuracy.
   - In preliminary standalone exploratory evaluations, a budget of `maxiter=100` was run on the fixed baseline, yielding 100.00% validation and 96.00% test accuracy.
   All reported metrics are traceable to these explicit optimizer budgets.

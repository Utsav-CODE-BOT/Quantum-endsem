# Research Methodology & Mathematical Formulation

## 1. Mathematical Formulation of Quantum Encoding & VQC

### 1.1 Quantum Angle Feature Encoding
Given an input image $\mathbf{I} \in \mathbb{R}^{28 \times 28}$, Principal Component Analysis fitted strictly on training data maps $\mathbf{I} \to \mathbf{z} \in \mathbb{R}^d$ ($d=4$). Rescaling maps each coordinate to $[0, \pi]$:

$$x_i = \pi \cdot \frac{z_i - z_{i,\min}}{z_{i,\max} - z_{i,\min}}$$

The feature map prepares the initial quantum state $|\psi(\mathbf{x})\rangle$ from the ground state $|0\rangle^{\otimes n}$:

$$|\psi(\mathbf{x})\rangle = U_{\Phi}(\mathbf{x}) |0\rangle^{\otimes n} = \left( \bigotimes_{i=1}^n R_y(x_i) \right) |0\rangle^{\otimes n}$$

where $R_y(\alpha) = \exp\left(-i \frac{\alpha}{2} Y\right)$. This encoding produces product states without entangling gates, maintaining zero circuit depth overhead for the feature map.

---

### 1.2 Variational Quantum Ansatz Architecture
The parameterized ansatz $U(\boldsymbol{\theta})$ transforms the state according to $L$ repeating layers:

$$|\Psi(\mathbf{x}, \boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |\psi(\mathbf{x})\rangle = \left[ \prod_{l=1}^L \left( U_{\text{ent}}^{(l)} \cdot \bigotimes_{q=1}^n \prod_{g \in \mathcal{G}} R_g(\theta_{l, q, g}) \right) \right] \left( \bigotimes_{q=1}^n \prod_{g \in \mathcal{G}} R_g(\theta_{0, q, g}) \right) |\psi(\mathbf{x})\rangle$$

where $\mathcal{G} \subseteq \{R_x, R_y, R_z\}$ represents the candidate rotation gate set, and $U_{\text{ent}}^{(l)}$ represents the entangling topology:
- **Linear Entanglement:** $\prod_{q=0}^{n-2} \text{CNOT}(q, q+1)$
- **Circular Entanglement:** $\text{CNOT}(n-1, 0) \prod_{q=0}^{n-2} \text{CNOT}(q, q+1)$
- **Pairwise Entanglement:** Alternating even pairs $\text{CNOT}(2k, 2k+1)$ and odd pairs $\text{CNOT}(2k+1, 2k+2)$.
- **None (Product Ansatz):** $U_{\text{ent}} = \mathbb{I}$

---

### 1.3 Parity Interpretation & Loss Function
Measurement in the computational basis yields bitstrings $\mathbf{b} \in \{0, 1\}^n$ with probability:

$$P(\mathbf{b} \mid \mathbf{x}, \boldsymbol{\theta}) = |\langle \mathbf{b} \mid \Psi(\mathbf{x}, \boldsymbol{\theta})\rangle|^2$$

The bitstring parity function maps the output to binary class labels $c \in \{0, 1\}$:

$$c(\mathbf{b}) = \left( \sum_{i=0}^{n-1} b_i \right) \pmod 2$$

The predicted class probabilities are:

$$P(Y = 0 \mid \mathbf{x}, \boldsymbol{\theta}) = \sum_{\mathbf{b}: c(\mathbf{b})=0} P(\mathbf{b} \mid \mathbf{x}, \boldsymbol{\theta}), \quad P(Y = 1 \mid \mathbf{x}, \boldsymbol{\theta}) = \sum_{\mathbf{b}: c(\mathbf{b})=1} P(\mathbf{b} \mid \mathbf{x}, \boldsymbol{\theta})$$

The classical optimizer minimizes cross-entropy loss over the training set $\mathcal{D}_{\text{train}}$:

$$\mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{k=1}^N \left[ y_k \ln P(Y = 1 \mid \mathbf{x}_k, \boldsymbol{\theta}) + (1 - y_k) \ln P(Y = 0 \mid \mathbf{x}_k, \boldsymbol{\theta}) \right]$$

using the derivative-free COBYLA (Constrained Optimization By Linear Approximation) algorithm.

---

## 2. Multi-Objective Evolutionary Architecture Search

### 2.1 Multi-Objective Fitness Formulation
To prevent the search from merely selecting maximally deep circuits that overfit or fail on noisy hardware, candidate architecture $A$ is scored using a normalized weighted-sum scalar objective (scalarization of accuracy, complexity, and noise robustness rather than explicit Pareto-frontier ranking algorithms such as NSGA-II):

$$F(A) = w_{\text{perf}} \cdot f_{\text{perf}}(A) - w_{\text{comp}} \cdot f_{\text{comp}}(A) + w_{\text{rob}} \cdot f_{\text{rob}}(A)$$

where:
1. **Performance Objective:** $f_{\text{perf}}(A) = \text{Accuracy}_{\text{val}}(A) \in [0, 1]$.
2. **Complexity Objective:**
   $$f_{\text{comp}}(A) = \frac{1}{2} \left( \frac{\text{depth}(A)}{\text{depth}_{\max}} + \frac{N_{2Q}(A)}{\text{CX}_{\max}} \right) \in [0, 1]$$
   with $\text{depth}_{\max} = 50$ and $\text{CX}_{\max} = 30$, where $N_{2Q}(A)$ denotes the count of two-qubit entangling gates ($CX, CZ$). This formulation directly penalizes two-qubit gates, which represent the dominant physical decoherence and gate-error bottleneck on NISQ hardware (typically $10\times$ higher error rate than single-qubit rotations).
3. **Robustness Objective:** 
   $$f_{\text{rob}}(A) = \begin{cases} 0.0 & \text{if } \text{Acc}_{\text{val}}(A) \le 0 \text{ or } \text{Acc}_{\text{noisy}}(A) \le 0 \text{ or missing} \\ \min\left(1, \max\left(0, \frac{\text{Acc}_{\text{noisy}}(A)}{\text{Acc}_{\text{val}}(A)}\right)\right) & \text{otherwise} \end{cases} \in [0, 1]$$
   where $\text{Acc}_{\text{noisy}}(A)$ is actively evaluated on the validation dataset using the standardized Medium Noise NISQ regime ($p_{1q}=0.10\%, p_{2q}=1.00\%, p_{\text{meas}}=1.50\%, 1024\text{ shots}$) via Qiskit Aer's density matrix simulator. Robustness represents the fraction of ideal validation accuracy retained under simulated NISQ noise, mathematically clamped to $[0, 1]$ to prevent distortion if noisy validation accuracy equals or exceeds ideal accuracy due to finite-shot sampling. If noise simulation fails or validation accuracy is zero or non-finite, robustness returns $0.0$, strictly preventing missing measurements from receiving artificial rewards.
4. **Default Objective Weights:** $w_{\text{perf}} = 0.50$, $w_{\text{comp}} = 0.30$, $w_{\text{rob}} = 0.20$ ($w_{\text{perf}} + w_{\text{comp}} + w_{\text{rob}} = 1.0$).

---

### 2.2 Genetic Operators & Search Dynamics

- **Chromosome Encoding:** Discrete gene vector representing $(\text{num\_layers}, \mathcal{G}, \text{entanglement\_type}, \text{entanglement\_freq})$.
- **Selection:** Binary Tournament Selection with tournament size $k=2$.
- **Crossover:** Uniform Gene Exchange with crossover probability $p_c = 0.70$.
- **Mutation:** Discrete perturbation with mutation rate $p_m = 0.30$:
  - Layer count mutation: $L \gets L \pm 1$ clamped to $[1, 6]$.
  - Rotation gate mutation: Random replacement or addition from $\{R_x, R_y, R_z\}$.
  - Entanglement topology mutation: Random swap between linear, circular, pairwise, none.
  - Entanglement frequency mutation: $k \gets k \pm 1$ clamped to $[1, 3]$.
- **Elitism:** The top individual ($n_{\text{elite}} = 1$) is guaranteed unconditional survival into the subsequent generation.
- **Evaluation Caching:** SHA-256 chromosome hashing ensures duplicate circuit architectures are retrieved from cache in $\mathcal{O}(1)$ time without duplicate quantum simulator training.

---

### 2.3 Scientific Discussion: Expressive Sufficiency, Trainability & Dataset Scope

1. **Analysis of Entangling Overhead and Expressive Sufficiency:**
   While PCA produces mutually orthogonal coordinate axes, this mathematical property does not, by itself, theoretically prove that a non-entangling or low-entangling quantum circuit has sufficient expressive capacity for classification. Rather, our justification is empirical, grounded in the measured performance of the discovered architecture: on this 360-sample, 4-feature binary dataset, the search-optimized model with minimal entangling operations ($CX = 1$ in `arch_4169ee9360`) achieved 90.00% test accuracy (88.33% validation accuracy in the standardized benchmark) while dramatically reducing exposure to two-qubit gate noise. Under our multi-objective fitness formulation, introducing unnecessary CX gates incurs an immediate complexity penalty ($\frac{N_{2Q}}{30}$) and amplifies simulated depolarizing noise, without offering marginal validation accuracy gains. The algorithm appropriately avoids unneeded entangling overhead on this specific task. Crucially, this task-specific result is an empirical finding and does not establish that entanglement is generally unnecessary in quantum machine learning or on higher-dimensional problems.

2. **Depth Degradation vs Formal Barren Plateaus:**
   The deeper VQC exhibited lower classification accuracy (92.00%) and required longer training time (34.95s - 51.74s) than the shallow (99.00%) and search-optimized (90.00%) architectures. This may reflect optimization difficulties, such as classical optimizer stagnation in a higher-dimensional parameter space (40 parameters) or overparameterization. However, gradient variance was not measured, so the presence of barren plateaus cannot be established from these results. To claim barren plateau behavior rigorously, one must measure gradient variance $\text{Var}_{\boldsymbol{\theta}}[\partial \mathcal{L}/\partial \theta_k]$ across circuit depths, initializations, and system sizes $n$.

3. **Optimization Budget Lineage:**
   To maintain computational tractability across generations during evolutionary search, candidate circuits were evaluated with a budget of `maxiter=25` COBYLA iterations, where `arch_4169ee9360` was initially discovered (`EXP-SEARCH-4169EE9360`: 90.00% val, 88.00% test). In the final standardized comparative benchmark (`EXP-BENCH-20261120`), all quantum models were retrained under a uniform budget of `maxiter=35` iterations (`arch_4169ee9360`: 88.33% val, 90.00% test; baseline VQC: 98.33% val, 95.00% test). Earlier standalone exploratory baseline training utilized `maxiter=100` (100.00% val, 96.00% test). Documenting these explicit optimizer budgets clarifies the performance lineage across experimental phases.

4. **Dataset Scope & Pilot Study Nature (360 Samples):**
   The experimental protocol uses a controlled 360-sample partition (200 train, 60 validation, 100 test). With 100 test samples, a single correct or incorrect prediction alters accuracy by exactly 1.0 percentage point. The 100.00% accuracy of classical SVM baselines demonstrates linear separability on this specific 4-dimensional projection, and does not imply full 10-class Fashion-MNIST classification. The results are presented as a controlled pilot experiment. All noise benchmarks represent classical simulations of hardware-inspired NISQ noise models using Qiskit Aer's density matrix backend, not physical quantum device executions.

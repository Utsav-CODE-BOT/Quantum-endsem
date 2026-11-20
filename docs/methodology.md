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
To prevent the search from merely selecting maximally deep circuits that overfit or fail on noisy hardware, candidate architecture $A$ is scored using a normalized scalar objective:

$$F(A) = w_{\text{perf}} \cdot f_{\text{perf}}(A) - w_{\text{comp}} \cdot f_{\text{comp}}(A) + w_{\text{rob}} \cdot f_{\text{rob}}(A)$$

where:
1. **Performance Objective:** $f_{\text{perf}}(A) = \text{Accuracy}_{\text{val}}(A) \in [0, 1]$.
2. **Complexity Objective:**
   $$f_{\text{comp}}(A) = \frac{1}{2} \left( \frac{\text{depth}(A)}{\text{depth}_{\max}} + \frac{N_{2Q}(A)}{\text{CX}_{\max}} \right) \in [0, 1]$$
   with $\text{depth}_{\max} = 50$ and $\text{CX}_{\max} = 30$, where $N_{2Q}(A)$ denotes the count of two-qubit entangling gates ($CX, CZ$). This formulation directly penalizes two-qubit gates, which represent the dominant physical decoherence and gate-error bottleneck on NISQ hardware (typically $10\times$ higher error rate than single-qubit rotations).
3. **Robustness Objective:** 
   $$f_{\text{rob}}(A) = \max\left(0, 1.0 - \frac{\text{Acc}_{\text{ideal}}(A) - \text{Acc}_{\text{noisy}}(A)}{\max(\text{Acc}_{\text{ideal}}(A), 10^{-4})}\right) \in [0, 1]$$
   where $\text{Acc}_{\text{noisy}}(A)$ is actively evaluated on the validation dataset using the standardized Medium Noise NISQ regime ($p_{1q}=0.10\%, p_{2q}=1.00\%, p_{\text{meas}}=1.50\%, 1024\text{ shots}$) via Qiskit Aer's density matrix simulator. If noise simulation fails for a candidate, the candidate fails evaluation ($F(A) = -1.0, \text{success} = \text{False}$) and is disqualified from evolutionary selection, strictly preventing missing measurements from receiving artificial rewards.
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

1. **Analysis of Zero CX Gates in the Optimal Circuit:**
   The evolutionary optimizer converged on an architecture without two-qubit CX gates ($CX = 0$). In this binary classification problem (Class 0 vs Class 9 compressed to 4 orthogonal PCA features), separable product states parameterized by single-qubit rotations provide sufficient expressive capacity to draw a high-accuracy separating hyper-surface. Under our multi-objective fitness formulation, introducing CX gates incurs an immediate complexity penalty ($\frac{N_{2Q}}{30}$) and amplifies simulated depolarizing noise, without offering marginal validation accuracy gains. The algorithm appropriately avoids unneeded entangling overhead.

2. **Depth Degradation vs Formal Barren Plateaus:**
   The deeper VQC exhibited lower classification accuracy (92.00%) than the shallow (99.00%) and search-optimized (96.00%) architectures. This may reflect optimization difficulties or overparameterization. However, gradient variance was not measured, so the presence of barren plateaus cannot be established from these results. To claim barren plateau behavior rigorously, one must measure gradient variance $\text{Var}_{\boldsymbol{\theta}}[\partial \mathcal{L}/\partial \theta_k]$ across circuit depths, initializations, and system sizes $n$.

3. **Dataset Scope (360 Samples):**
   The experimental protocol uses a controlled 360-sample partition (200 train, 60 validation, 100 test). The 100.00% accuracy of classical SVM baselines demonstrates linear separability on this specific 4-dimensional projection, and does not imply full 10-class Fashion-MNIST classification.

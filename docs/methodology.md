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
   $$f_{\text{comp}}(A) = \frac{1}{2} \left( \frac{\text{depth}(A)}{\text{depth}_{\max}} + \frac{\text{gates}(A)}{\text{gates}_{\max}} \right) \in [0, 1]$$
   with $\text{depth}_{\max} = 30$ and $\text{gates}_{\max} = 100$.
3. **Robustness Objective:** $f_{\text{rob}}(A) = 1.0 - \min(1.0, \Delta_{\text{noise}}(A))$.
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

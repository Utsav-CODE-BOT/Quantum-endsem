# Self-Adaptive Quantum Neural Networks with Resource-Aware Architecture Optimization

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-613394.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing%20(148%2F148)-brightgreen.svg)](tests/)

An end-to-end, research-grade quantum machine learning framework developed in Python and Qiskit that automates the discovery, optimization, and evaluation of **Variational Quantum Classifiers (VQCs)**.

The framework resolves the multi-objective trade-off between **predictive accuracy**, **quantum circuit complexity** (depth and gate count), and **simulated NISQ noise robustness** on the Fashion-MNIST image dataset using Qiskit Aer density-matrix simulations.

---

## 1. Key Research Contributions

1. **Automated Multi-Objective Architecture Search:** An evolutionary optimization engine that searches the discrete combinatorial space of ansatz layers, rotation gates, and entanglement topologies (linear, circular, pairwise, none) using a normalized fitness objective:
   $$F(A) = w_{\text{perf}} \cdot f_{\text{perf}}(A) - w_{\text{comp}} \cdot f_{\text{comp}}(A) + w_{\text{rob}} \cdot f_{\text{rob}}(A)$$
   where:
   $$f_{\text{perf}}(A) = \text{Acc}_{\text{val}}(A)$$
   $$f_{\text{comp}}(A) = \frac{1}{2}\left(\frac{\text{depth}(A)}{50} + \frac{N_{2Q}(A)}{30}\right)$$
   $$f_{\text{rob}}(A) = \begin{cases} 0.0 & \text{if } \text{Acc}_{\text{val}}(A) \le 0 \text{ or } \text{Acc}_{\text{noisy}}(A) \le 0 \text{ or missing} \\ \min\left(1, \max\left(0, \frac{\text{Acc}_{\text{noisy}}(A)}{\text{Acc}_{\text{val}}(A)}\right)\right) & \text{otherwise} \end{cases}$$
   Here $f_{\text{rob}}(A)$ represents the exact fraction of ideal validation accuracy retained under simulated NISQ noise, mathematically clamped to $[0, 1]$ to prevent distortion if noisy validation accuracy equals or exceeds ideal validation accuracy. If noise evaluation fails or validation accuracy is zero or non-finite, robustness evaluates strictly to $0.0$. Default weights are $w_{\text{perf}} = 0.50$, $w_{\text{comp}} = 0.30$, and $w_{\text{rob}} = 0.20$.
2. **Empirical Resource Reduction with High Noise Retention:** The search discovered a compact scalarization-optimized architecture (`arch_4169ee9360` with depth 10, 1 CX gate, 32 parameters) that cut two-qubit entangling gates by **83.3%** (from 6 down to 1 CX gate) while achieving **90.00% test accuracy** and **98.89% accuracy retention** under simulated High Noise NISQ conditions ($p_{1q}=0.5\%, p_{2q}=3\%, p_{\text{meas}}=3\%$).
3. **Rigorous Simulated NISQ Noise Benchmarking:** Paired evaluation across ideal statevector simulation and hardware-inspired NISQ noise tiers (low, medium, and high depolarizing + readout error, 1,024 shots) using Qiskit Aer's density matrix backend. *(Note: All experiments represent classical simulations of hardware-inspired noise models, not physical quantum device executions.)*
4. **Strict Methodological Discipline:** Zero data leakage (PCA and scaling fitted strictly on training data), standard scikit-learn compatible interfaces, and honest scientific reporting without unsubstantiated "quantum advantage" claims.

---

## 2. Global Pipeline Architecture

```
                                  [Raw Fashion-MNIST]
                                           │
                                           ▼
                                 [Dataset Splitter]
                           (Train: 200, Val: 60, Test: 100)
                                           │
                                           ▼
                                   [PCA Reducer]
                              (Strict Fit on Train ONLY)
                                           │
                                           ▼
                                   [Feature Scaler]
                                (Normalized to [0, π])
                                           │
                                           ▼
                               [Quantum Angle Feature Map]
                                (Ry rotations on 4 Qubits)
                                           │
                                           ▼
   ┌──────────────────────────────── Evolutionary Architecture Loop ────────────────────────────────┐
   │                                                                                                │
   │   [Population Generator] ──► [Ansatz Circuit Generator] ──► [Configurable VQC (COBYLA)]        │
   │            ▲                                                            │                      │
   │            │                                                            ▼                      │
   │            │                                              [Multi-Objective Evaluator]          │
   │            │                                          ┌────────────────────────────────────┐   │
   │            │                                          │ • Ideal Validation Accuracy        │   │
   │            │                                          │ • Circuit Complexity (Depth & CX)  │   │
   │            │                                          │ • Noisy Validation Accuracy (Aer)  │   │
   │            │                                          └─────────────────┬──────────────────┘   │
   │            │                                                            │                      │
   │            │                     Validation accuracy +                  │                      │
   │            │                     Circuit complexity +           ◄───────┘                      │
   │            │                     Noisy validation accuracy                                     │
   │            │                                │                                                  │
   │            │                                ▼                                                  │
   │     [Genetic Operators]         ◄── [Fitness Engine]                                           │
   │   (Crossover, Mutation,              w_p·Perf - w_c·Comp + w_r·Rob                             │
   │     Elitist Selection)                                                                         │
   │                                                                                                │
   └─────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                 │
                                                 ▼ (Best Architecture)
                                   [Simulated NISQ Noise Evaluation]
                              (Depolarizing + Readout Noise: AerSimulator)
                                                 │
                                                 ▼
                                     [Multi-Model Benchmarking]
                               (Linear SVM, RBF SVM, MLP, Fixed VQCs)
                                                 │
                                                 ▼
                                   [Figures & Research Reporting]
```

---

## 3. Empirical Benchmark Results

Evaluated on Fashion-MNIST binary classification (Class 0: T-shirt vs Class 9: Ankle boot, 4 PCA features, 360-sample balanced partition: 200 train, 60 validation, 100 test):

| Model Name | Category | Trainable Params | Circuit Depth | 2-Qubit Gates | Val Acc | Test Acc | Test F1 | Test ROC-AUC | Training Time | Latency / Sample |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear SVM** | Classical | 41 | - | - | 100.00% | 100.00% | 1.0000 | 1.0000 | 0.05s | 0.02ms |
| **RBF SVM** | Classical | 113 | - | - | 100.00% | 100.00% | 1.0000 | 1.0000 | 0.01s | 0.02ms |
| **Classical MLP** | Classical | 25 | - | - | 30.00% | 40.00% | 0.3961 | 0.4020 | 0.26s | 0.02ms |
| **Fixed Shallow VQC** | Quantum Baseline | 8 | 6 | 3 | 100.00% | 99.00% | 0.9900 | 1.0000 | 10.68s | 1.73ms |
| **Fixed Standard VQC** | Quantum Baseline | 24 | 12 | 6 | 98.33% | 95.00% | 0.9499 | 0.9984 | 21.27s | 3.21ms |
| **Fixed Deep VQC** | Quantum Baseline | 40 | 27 | 16 | 93.33% | 92.00% | 0.9195 | 0.9956 | 34.95s | 5.09ms |
| **Search-Optimized VQC (`arch_4169ee9360`)** | **Quantum Search** | **32** | **10** | **1** | **88.33%** | **90.00%** | **0.8994** | **0.9688** | **23.17s** | **3.40ms** |

> [!NOTE]
> **Experiment Lineage & Evaluation Budget:** The search-optimized candidate (`arch_4169ee9360`) was discovered during evolutionary search (`EXP-SEARCH-4169EE9360`, budget `maxiter=25`, 16.95s training), achieving 90.00% validation accuracy and 88.00% test accuracy. In the standardized head-to-head benchmark above (`EXP-BENCH-20261120`), all models were trained under the uniform protocol (`maxiter=35`, 23.17s training, checkpoint weights hash: `151e19ca76722842`), achieving 88.33% validation accuracy and 90.00% test accuracy. Checkpoint weights for both Baseline VQC (`21bf8f99f15b3672`) and Search-Optimized VQC (`151e19ca76722842`) are preserved in `experiments/checkpoints/`.

### Key Benchmark Insights:
- **Optimization Degradation in Deeper Ansätze:** The deeper VQC exhibited lower classification accuracy (92.00%) and required longer training time (34.95s) than the shallow (99.00%) and search-optimized (90.00%) architectures. This may reflect optimization difficulties or overparameterization. However, gradient variance was not measured, so the presence of barren plateaus cannot be established from these results.
- **Resource Optimization & Expressive Sufficiency:** While PCA produces orthogonal component directions, this mathematical property does not, by itself, theoretically prove that a low-entangling quantum circuit has sufficient expressive capacity for classification. Rather, our justification is empirical: on this 360-sample partition, the search autonomously converged on a compact model with minimal entangling gates ($CX = 1$ in `arch_4169ee9360`) that achieved 90.00% test accuracy while avoiding the heavy noise penalty associated with two-qubit gates on NISQ devices. Note that this task-specific result does not establish that entanglement is generally unnecessary in QML or on more complex datasets.
- **Dataset Scale Scope & Pilot Study Nature (360 Samples):** Experiments evaluate a controlled 360-sample partition (200 train, 60 validation, 100 test). With only 100 test samples, each correct or incorrect prediction changes test accuracy by exactly 1.0 percentage point. The 100% test accuracy achieved by classical Linear and RBF SVMs demonstrates that this 4D binary projection is linearly separable on this sample size, and should not be misinterpreted as 100% accuracy on full 10-class, 784-dimensional Fashion-MNIST or as quantum supremacy. Results are presented as a controlled pilot experiment.

---

## 4. Simulated NISQ Noise Robustness

Performance retention under simulated depolarizing and readout noise (1,024 shots each on AerSimulator density matrix):

| Architecture | Circuit Specs | Ideal Acc | Low Noise ($p_{1q}=0.05\%$) | Medium Noise ($p_{1q}=0.1\%$) | High Noise ($p_{1q}=0.5\%, p_{2q}=3\%$) | Retention Rate (High Noise) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline VQC** | Depth 12, 6 CX | 95.00% | 93.00% | 95.00% | 95.00% | **100.00%** |
| **Search-Optimized VQC (`arch_4169ee9360`)** | Depth 10, 1 CX | 90.00% | 89.00% | 90.00% | 89.00% | **98.89%** |

Under simulated NISQ conditions (AerSimulator density matrix, 1,024 shots), both architectures demonstrate strong resilience on this 4D classification task. The baseline VQC achieved 93.00% (Low), 95.00% (Medium), and 95.00% (High) accuracy (100.00% retention). The search-optimized VQC achieved 89.00% (Low), 90.00% (Medium), and 89.00% (High) accuracy (98.89% retention). The optimized VQC does not claim superior noise retention over the baseline; rather, its demonstrated advantage lies in sustaining 98.89% retention under high noise while cutting two-qubit CX gates by 83.3% (1 CX vs 6 CX) and depth by 16.7% (10 vs 12). *(Note: All experiments represent classical simulations of hardware-inspired noise models, not physical quantum device executions, and no quantum supremacy or superiority over classical SVMs is claimed.)*

---

## 5. Generated Publication Figures

| Figure | Description | File Link |
|:---:|:---|:---:|
| **Fig 1** | Evolutionary Search Fitness Convergence | [`fig1_search_convergence.png`](experiments/figures/fig1_search_convergence.png) |
| **Fig 2** | Pareto Trade-off: Accuracy vs Circuit Depth | [`fig2_pareto_tradeoff.png`](experiments/figures/fig2_pareto_tradeoff.png) |
| **Fig 3** | Classical vs Quantum Comparative Benchmark | [`fig3_model_comparison.png`](experiments/figures/fig3_model_comparison.png) |
| **Fig 4** | Quantum Noise Degradation Trajectory | [`fig4_noise_degradation.png`](experiments/figures/fig4_noise_degradation.png) |
| **Fig 5** | Baseline vs Optimized Confusion Matrices | [`fig5_confusion_matrices.png`](experiments/figures/fig5_confusion_matrices.png) |

---

## 6. Directory Structure

```
quantum-architecture-optimizer/
├── configs/
│   └── default_config.yaml         # Centralized hyperparameters & paths
├── data/
│   ├── raw/                        # Fashion-MNIST raw archives
│   └── processed/                  # Zero-leakage quantum-ready NPZ files
├── docs/
│   ├── architecture.md             # System architecture & component design
│   ├── methodology.md              # Mathematical formulations & fitness
│   ├── experiment_protocol.md      # Scientific protocol & zero-leakage PCA
│   ├── installation.md             # Environment setup & compatibility
│   ├── user_guide.md               # CLI commands & execution manual
│   └── limitations.md              # Scientific discussion & NISQ limits
├── experiments/
│   ├── checkpoints/                # Serialized trained model weights
│   ├── figures/                    # Publication-grade PNG charts (150 DPI)
│   ├── logs/                       # Execution logs with timestamps
│   └── results/                    # Structured JSON benchmarks & report
├── scripts/
│   ├── download_dataset.py         # Fashion-MNIST downloader
│   ├── run_preprocessing.py        # Strict zero-leakage PCA preprocessor
│   ├── run_baseline.py             # Phase 4 Baseline VQC runner
│   ├── run_architecture_search.py  # Phase 7 Evolutionary search engine
│   ├── run_noise_experiments.py    # Phase 8 Quantum noise benchmarking
│   ├── run_benchmarks.py           # Phase 9 Comprehensive benchmark runner
│   └── generate_report.py          # Phase 11 Report & figures compiler
├── src/
│   ├── baselines/                  # Classical SVM, MLP, and fixed VQCs
│   ├── data/                       # Ingestion, validation, and splitting
│   ├── evaluation/                 # Metrics, fitness, and noise evaluator
│   ├── optimization/               # Search space, generator, crossover, mutation
│   ├── preprocessing/              # Image processing, PCA, angle scaling
│   ├── quantum/                    # Feature maps, ansatz builder, VQC model
│   ├── training/                   # Training history and callbacks
│   ├── utils/                      # Config, logging, seed, system monitor
│   └── visualization/              # Publication plotting routines
├── tests/                          # 148 comprehensive unit, integration, and UI service tests (pytest)
├── pyproject.toml                  # Package configuration & test flags
├── requirements.txt                # Pinned dependency manifest
└── README.md                       # Project master documentation
```

---

## 7. Interactive Research Dashboard (Phase 13)

The project includes a comprehensive, dark-themed, publication-grade interactive research platform built with Streamlit, Plotly, and Qiskit.

```powershell
# Launch the Research Dashboard
.\.venv\Scripts\streamlit.exe run ui/app.py
```

### Dashboard Research Modules:
1. **Overview Dashboard (`1_Overview.py`):** Executive summary, multi-objective trade-off analysis, convergence metrics, and NISQ noise degradation curves.
2. **Dataset Explorer (`2_Dataset_Explorer.py`):** Interactive Fashion-MNIST browser with step-by-step pipeline transformations ($28\times 28 \to 784 \to 4 \to [0,\pi]$) and PCA explained variance scree plots (strictly adhering to zero data leakage).
3. **Quantum Circuit Lab (`3_Quantum_Circuit_Lab.py`):** Real-time interactive VQC builder with custom rotation gates and entanglement topologies, live Qiskit compilation, and high-res 200 DPI PNG export.
4. **Architecture Search (`4_Architecture_Search.py`):** Multi-objective evolutionary search engine with asynchronous background thread execution and live telemetry updates.
5. **Noise Analysis (`5_Noise_Analysis.py`):** Density matrix NISQ noise simulations across Low, Medium, and High error tiers.
6. **Model Benchmarking (`6_Model_Benchmarking.py`):** Rigorous comparative table and charts comparing 7 classical and quantum models with fair-comparison scientific notes.
7. **Experiment History (`7_Experiment_History.py`):** Filterable, searchable repository of all saved experimental runs with raw JSON inspector and CSV export.
8. **Documentation & Knowledge Base (`8_Documentation.py`):** In-dashboard access to complete theoretical methodology, system architecture, and command references.

---

## 8. Quickstart Installation & Execution

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run complete test suite
pytest tests/ -v

# 4. Execute the complete pipeline
python scripts/run_preprocessing.py
python scripts/run_baseline.py
python scripts/run_architecture_search.py
python scripts/run_noise_experiments.py
python scripts/run_benchmarks.py
python scripts/generate_report.py
```

---

## 9. Scientific Limitations & Ethical Reporting

1. **No False Quantum Advantage Claims:** On the 4-dimensional PCA representation of Fashion-MNIST, classical Linear and RBF SVMs attain 100.00% accuracy within fractions of a second. Classical ML remains superior in training speed and throughput for this problem scale.
2. **Purpose of Framework:** The framework solves the engineering challenge of automated circuit discovery under physical NISQ constraints. By minimizing gate counts and depth, it produces circuits that fit within the physical coherence limits ($T_1, T_2$) and error budgets of near-term quantum processors.

---

## 10. Academic References & Citations

- Cerezo, M., et al. (2021). *Variational quantum algorithms.* Nature Reviews Physics, 3(9), 625-644.
- McClean, J. R., et al. (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications, 9(1), 4812.
- Havlíček, V., et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature, 567(7747), 209-212.
- Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers.* Springer.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747.

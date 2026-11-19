# Phase 13 User Guide: Interactive Research Dashboard

**Framework:** Quantum Architecture Optimizer Research Platform  
**Target Audience:** Researchers, Quantum ML Engineers, Peer Reviewers  

---

## 1. Quick Launch

Launch the interactive dashboard from the project workspace root using the activated virtual environment:

```powershell
cd C:\Users\ud127\.gemini\antigravity\scratch\quantum-architecture-optimizer
.\.venv\Scripts\streamlit.exe run ui/app.py
```

Open your web browser at `http://localhost:8501`.

---

## 2. Navigating the 8 Research Modules

The application features a responsive sidebar with live backend health telemetry, host resource status, and 8 dedicated research pages:

### Page 1: Overview Dashboard (`1_Overview.py`)
- **Purpose:** High-level summary of the research study and key empirical discoveries.
- **Key Visualizations:**
  - Executive KPI Cards: Test Accuracy (95.0%), Circuit Depth Reduction (-58.3%), Two-Qubit CX Gates (0), High Noise Retention (100%).
  - Multi-Objective Pareto Frontier: Plotly interactive scatter showing all evaluated architectures in accuracy vs. depth space with non-dominated Pareto marking.
  - Fitness Convergence: Evolutionary progress across generations.
  - Model Performance: Side-by-side comparison with classical baselines (Linear SVM, RBF SVM, MLP).
  - Noise Resilience: Accuracy retention curves across Low, Medium, and High NISQ noise tiers.

### Page 2: Dataset Explorer (`2_Dataset_Explorer.py`)
- **Purpose:** Step-by-step verification of data ingestion, preprocessing, dimensionality reduction, and zero data leakage.
- **Features:**
  - Interactive sample browser: Slider to choose any Fashion-MNIST sample index.
  - 4-Stage Pipeline Walkthrough:
    1. Raw Image (28 $\times$ 28 grayscale pixel grid).
    2. Flattened & Normalized Vector (784 features in $[0, 1]$).
    3. PCA Dimensionality Reduction ($784 \to 4$ orthogonal components).
    4. Quantum Angle Encoding (feature scaling to $[0, \pi]$ radians).
  - PCA Explained Variance Scree Plot: Visualizing individual and cumulative variance captured by each principal component.
  - Dataset Summary Table: Train/Val/Test partitions ($840 / 180 / 180$).

### Page 3: Quantum Circuit Lab (`3_Quantum_Circuit_Lab.py`)
- **Purpose:** Real-time quantum circuit design, compilation, and gate-level resource profiling.
- **Features:**
  - Predefined Presets: One-click loading of "Search-Optimized VQC", "Baseline Standard VQC", "Fixed Shallow VQC", and "Fixed Deep VQC".
  - Interactive Parameter Controls: Qubit count ($2 \dots 8$), Layer depth ($1 \dots 4$), Rotation gates ($R_X, R_Y, R_Z$), Entanglement topology (`linear`, `circular`, `full`, `none`), and Two-qubit gate type (`cx`, `cz`).
  - Real-Time Qiskit Compilation: Instantly calculates decomposed circuit depth, total gate count, single-qubit vs. two-qubit gate distribution, and parameter count.
  - High-Resolution Figure Export: Matplotlib dark-themed circuit diagram with a one-click **"Download Circuit PNG (200 DPI)"** button.
  - Unicode Text Fallback: Collapsible ASCII diagram for terminal-grade visualization.

### Page 4: Architecture Search (`4_Architecture_Search.py`)
- **Purpose:** Configure, launch, and monitor evolutionary multi-objective architecture optimization with live telemetry.
- **Features:**
  - Evolutionary Hyperparameters: Population size, Generation count, Mutation probability, Crossover probability, Tournament size, and Random seed.
  - Multi-Objective Fitness Weight Sliders: Performance weight ($w_{\text{perf}}$), Complexity weight ($w_{\text{comp}}$), and Noise robustness weight ($w_{\text{rob}}$).
  - Background Execution: Runs asynchronously on a detached thread without freezing the Streamlit interface.
  - Live Telemetry: Best fitness, current generation progress bar, elapsed execution time, and real-time convergence charts.
  - Evaluated Candidate Table: Tabular breakdown of all explored circuit candidates.

### Page 5: Noise Analysis (`5_Noise_Analysis.py`)
- **Purpose:** Rigorous simulation of NISQ device noise using Qiskit Aer density matrix simulation.
- **Features:**
  - Noise Model Explorer: Inspect physical error parameters across Low, Medium, and High NISQ noise regimes (single-qubit depolarizing error, CX gate error, and readout measurement error).
  - Empirical Noise Degradation Table: Compare the Search-Optimized VQC against the Baseline Standard VQC under increasing noise levels.
  - Live Noise Simulation: Run interactive tests on custom circuit architectures with configurable noise parameters and sample sizes.

### Page 6: Model Benchmarking (`6_Model_Benchmarking.py`)
- **Purpose:** Cross-paradigm comparative analysis of 7 evaluated classical and quantum models.
- **Models Benchmarked:**
  1. Linear SVM
  2. RBF Kernel SVM
  3. Classical Multi-Layer Perceptron (MLP)
  4. Fixed Shallow VQC (1 layer)
  5. Standard Baseline VQC (2 layers, linear entanglement)
  6. Fixed Deep VQC (4 layers, circular entanglement)
  7. Search-Optimized VQC (arch_6327d7bcb8)
- **Features:**
  - Full Metrics Table: Test accuracy, circuit depth, CX gates, trainable parameters, training runtime, and inference latency.
  - Barren Plateau & Trainability Analysis: Discussion of gradient variance scaling and vanishing gradients in deep ansatze.
  - Fair Comparison Notice: Clear scientific caveats regarding classical CPU execution vs. simulated NISQ performance.

### Page 7: Experiment History (`7_Experiment_History.py`)
- **Purpose:** Persistent archive and audit trail of all saved experimental runs.
- **Features:**
  - Category Filtering: Filter runs by Quantum Baseline, Quantum Search, Evolutionary Optimization, or Noise Simulation.
  - Search Filter: Search by run title or experiment ID.
  - Raw JSON Inspector: Expand and inspect full serialized experiment dictionaries.
  - One-Click CSV Export: Download the complete experimental run table as a CSV spreadsheet.

### Page 8: Documentation & Knowledge Base (`8_Documentation.py`)
- **Purpose:** In-dashboard access to complete project technical documentation and mathematical formulations.
- **Features:**
  - Document Selector: Switch between System Architecture, Theoretical Methodology, Experimental Protocol, NISQ Limitations, Research Report, and User Guide.
  - In-Document Search: Real-time keyword filtering with line-number excerpts.
  - Markdown Download: Export any documentation file with one click.
  - Quick CLI Command Reference: Table of command-line scripts for data preparation, model training, and search execution.
  - System Environment Telemetry: Python, Qiskit, and OS version summary.

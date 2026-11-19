# Installation & Environment Setup Guide

## 1. System Requirements

- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS 12+ (x86_64 or arm64)
- **Python Version:** Python 3.10 or 3.11 (Python 3.11 recommended; Python 3.12+ has known wheel incompatibilities with specific Qiskit Aer versions).
- **RAM:** Minimum 8 GB (16 GB recommended for multi-candidate simulation).
- **Disk Space:** ~3 GB free space (for virtual environment, PyTorch wheels, and datasets).

---

## 2. Pinned Quantum Machine Learning Stack

To prevent breaking API changes and dependency conflicts, the following verified version stack is strictly pinned:

| Package | Pinned Version | Role / Compatibility Justification |
|:---|:---:|:---|
| `qiskit` | `1.4.2` | Core quantum circuit representation and transpiler |
| `qiskit-aer` | `0.16.4` | High-performance C++ quantum noise simulator |
| `qiskit-machine-learning` | `0.8.4` | VQC and SamplerQNN algorithms (strictly compatible with Qiskit 1.4.2) |
| `qiskit-algorithms` | `0.4.0` | Classical optimizers (COBYLA, SPSA, L_BFGS_B) |
| `scikit-learn` | `1.6.1` | PCA, feature scaling, SVM/MLP baselines, classification metrics |
| `torch` / `torchvision` | `2.15.0+cpu` | Fashion-MNIST dataset download and tensor utilities |
| `numpy` | `1.26.4` | Matrix computations and statevector processing |
| `matplotlib` | `3.10.1` | Publication-grade figure and plot generation |
| `pytest` | `9.2.0` | Automated unit and integration test suite |

> **Critical Compatibility Note:** `qiskit-machine-learning >= 0.9.0` enforces `qiskit >= 2.0.0`, whereas Qiskit 2.0 removes legacy Sampler primitives required by many NISQ evaluation routines. Using `qiskit==1.4.2` with `qiskit-machine-learning==0.8.4` guarantees maximum stability and prevents deprecation crashes.

---

## 3. Step-by-Step Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/quantum-architecture-optimizer.git
cd quantum-architecture-optimizer
```

### Step 2: Create Isolated Virtual Environment
```bash
python -m venv .venv
```
Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Smoke Tests & Verification
Verify that the entire quantum circuit generation, encoding, and optimization stack functions cleanly:
```bash
pytest tests/ -v
```
All tests should pass without errors.

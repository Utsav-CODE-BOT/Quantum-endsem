# Interactive Research Dashboard Setup & Quickstart

This document provides setup instructions and launch commands for the **Quantum Architecture Optimizer Interactive Research Dashboard (Phase 13)**.

---

## Prerequisites

- **Operating System:** Windows, macOS, or Linux
- **Python:** Python 3.11.x (installed in project `.venv`)
- **Quantum Stack:** Qiskit 1.4.2, Qiskit Aer 0.16.4, Qiskit Machine Learning 0.8.4
- **Dashboard Stack:** Streamlit >= 1.35.0, Plotly >= 5.20.0, Matplotlib >= 3.7.0

---

## Step 1: Virtual Environment Verification

Ensure the local virtual environment `.venv` is intact and active:

```powershell
cd C:\Users\ud127\.gemini\antigravity\scratch\quantum-architecture-optimizer

# Verify python executable
.\.venv\Scripts\python.exe --version
# Expected: Python 3.11.x

# Verify streamlit executable
.\.venv\Scripts\streamlit.exe --version
# Expected: Streamlit, version 1.47.x
```

---

## Step 2: Launching the Dashboard

Launch the Streamlit application using the local virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py
```

Streamlit will print the local network URL in your terminal:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://...:8501
```

Open your browser to `http://localhost:8501`.

---

## Step 3: Headless Mode (Optional / Server Deployments)

For headless or remote server environments, run:

```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py --server.headless=true --server.port=8501
```

---

## Step 4: Running Automated Tests

To run the entire test suite (core quantum machine learning backend + UI services and integration tests):

```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

To run UI-specific test modules:

```powershell
.\.venv\Scripts\pytest.exe tests/test_ui_services.py tests/test_ui_validation.py tests/test_ui_integration.py -v
```

---

## Key Dashboard Features

1. **Overview:** Executive summary, multi-objective Pareto front, convergence metrics.
2. **Dataset Explorer:** Fashion-MNIST sample browser, zero-leakage 4-stage pipeline inspection, PCA scree plots.
3. **Quantum Circuit Lab:** Interactive parametric VQC builder, real-time Qiskit compilation, high-res 200 DPI PNG export.
4. **Architecture Search:** Asynchronous evolutionary search engine with live telemetry and candidate ranking.
5. **Noise Analysis:** NISQ noise tiers (Low, Medium, High) with Qiskit Aer simulation and error degradation analysis.
6. **Model Benchmarking:** Classical vs. Quantum comparison table across 7 models with fair comparison warnings.
7. **Experiment History:** Searchable archive of completed runs, JSON payload inspector, and CSV export.
8. **Documentation:** Full system architecture, theoretical methodology, and user guides rendered inside the application.

# Phase 13 Implementation Plan: Quantum Architecture Optimizer Research Dashboard

## 1. UI Architecture Overview

The **Quantum Architecture Optimizer** dashboard is designed as a modular, high-performance research platform built on **Streamlit** and **Plotly**, connecting directly to the existing quantum machine learning backend.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Streamlit Presentation Layer (UI)                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │  Sidebar Navigation: Overview | Dataset | Circuit Lab |        │   │
│   │  Search | Noise | Benchmarks | Experiments | Docs              │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       UI Service / Adapter Layer                       │
│  ├── dataset_service.py    (Fashion-MNIST loader, samples, PCA)        │
│  ├── circuit_service.py    (Qiskit circuit compiler, draw MPL/text)    │
│  ├── search_service.py     (Threaded Evolutionary Search & monitoring) │
│  ├── noise_service.py      (Qiskit Aer NoiseModel & paired evaluation) │
│  ├── benchmark_service.py  (7-model classical vs quantum records)      │
│  └── experiment_service.py (JSON experiment loader, compare, export)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Existing Quantum Research Backend                    │
│  ├── src.data             (FashionMNISTLoader, DatasetSplitter)        │
│  ├── src.preprocessing    (PreprocessingPipeline, PCAReducer)          │
│  ├── src.quantum          (ConfigurableVQC, AnsatzBuilder, FeatureMap) │
│  ├── src.optimization     (EvolutionarySearchEngine, Space, Generator) │
│  ├── src.evaluation       (NoiseEvaluator, FitnessCalculator, Metrics) │
│  └── src.baselines        (ClassicalSVM, ClassicalMLP, FixedVQCs)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Persisted Research Artifacts                      │
│  ├── data/processed/fashion_mnist_quantum_ready.npz                    │
│  ├── data/metadata/preprocessing_metadata.json                         │
│  ├── experiments/checkpoints/baseline_vqc.pkl                          │
│  ├── experiments/results/*.json (baseline, best_arch, search, noise)   │
│  └── experiments/figures/*.png                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Backend Integration Strategy

1. **Direct Backend Reuse:**
   No quantum algorithms, model architectures, or fitness formulations will be rewritten. The UI service layer wraps existing classes (`FashionMNISTLoader`, `PreprocessingPipeline`, `ArchitectureGenerator`, `VQCCircuitGenerator`, `EvolutionarySearchEngine`, `NoiseEvaluator`).
2. **Read-Only Instant Mode:**
   All pages will default to loading existing saved results (`benchmark_comparison.json`, `search_history.json`, `best_architecture.json`, `noise_experiment_results.json`) to provide sub-second page loads without requiring simulator execution.
3. **Interactive Simulation Mode:**
   When the user initiates a circuit build, noise test, or evolutionary search, the service layer will invoke the backend dynamically and update the UI in real time.
4. **Data Isolation & Zero Leakage:**
   Test data is strictly segregated from validation-based evolutionary search. Preprocessing uses the pre-fitted PCA pipeline.

---

## 3. File Inventory

### 3.1 Existing Files Reused (Read-Only / Unmodified)
- `src/data/dataset_loader.py`
- `src/data/dataset_splitter.py`
- `src/preprocessing/preprocessing_pipeline.py`
- `src/preprocessing/pca_reducer.py`
- `src/preprocessing/feature_scaler.py`
- `src/quantum/architecture_schema.py`
- `src/quantum/ansatz_builder.py`
- `src/quantum/feature_encoding.py`
- `src/quantum/circuit_generator.py`
- `src/quantum/vqc_model.py`
- `src/quantum/simulator.py`
- `src/optimization/architecture_space.py`
- `src/optimization/architecture_generator.py`
- `src/optimization/evolutionary_search.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/fitness.py`
- `src/evaluation/noise_evaluator.py`
- `src/evaluation/classification_metrics.py`
- `src/baselines/classical_svm.py`
- `src/baselines/classical_nn.py`
- `src/baselines/fixed_vqc.py`
- `data/processed/fashion_mnist_quantum_ready.npz`
- `experiments/results/*.json`
- `experiments/figures/*.png`

### 3.2 Existing Files That Must NOT Be Modified
Core algorithm files (`src/quantum/ansatz_builder.py`, `src/optimization/crossover.py`, `src/optimization/mutation.py`, `src/evaluation/fitness.py`) will remain strictly untouched to preserve scientific validity.

### 3.3 New Files to Create
```
ui/
├── app.py                         # Application entrypoint & layout
├── styles/
│   └── theme.css                  # Custom research dark theme CSS
├── components/
│   ├── sidebar.py                 # Navigation sidebar with backend status
│   ├── metric_cards.py            # KPI metric card widgets
│   ├── charts.py                  # Interactive Plotly chart builders
│   └── circuit_viewer.py          # Qiskit circuit renderer (MPL & Text)
├── services/
│   ├── dataset_service.py         # Data exploration & preprocessing adapter
│   ├── circuit_service.py         # Quantum circuit generation & metrics adapter
│   ├── search_service.py          # Evolutionary search background manager
│   ├── noise_service.py           # Noise simulation adapter
│   ├── benchmark_service.py       # Comparative benchmarking adapter
│   └── experiment_service.py      # Experiment persistence & comparison
├── utils/
│   ├── formatting.py              # Number and string formatters
│   ├── validation.py              # Configuration validation helpers
│   └── state.py                   # Streamlit session state manager
├── pages/
│   ├── 1_Overview.py              # Page 1: Overview Dashboard
│   ├── 2_Dataset_Explorer.py      # Page 2: Fashion-MNIST Explorer
│   ├── 3_Quantum_Circuit_Lab.py   # Page 3: Quantum Circuit Lab
│   ├── 4_Architecture_Search.py   # Page 4: Evolutionary Search Manager
│   ├── 5_Noise_Analysis.py        # Page 5: Quantum Noise Analysis
│   ├── 6_Model_Benchmarking.py    # Page 6: Model Benchmarks
│   ├── 7_Experiment_History.py    # Page 7: Experiment History & Comparison
│   └── 8_Documentation.py         # Page 8: Project Documentation
tests/
├── test_ui_services.py            # Unit tests for UI adapter services
├── test_ui_validation.py          # Tests for input validation and state
└── test_ui_integration.py         # End-to-end integration tests
docs/
├── PHASE_13_BACKEND_AUDIT.md       # Audit deliverable
├── PHASE_13_IMPLEMENTATION_PLAN.md # This plan
├── PHASE_13_UI_ARCHITECTURE.md     # Detailed UI architectural specification
├── PHASE_13_TEST_REPORT.md         # UI test report
├── PHASE_13_USER_GUIDE.md          # User manual for dashboard
└── UI_SETUP.md                     # Quickstart launch guide
```

---

## 4. Dependencies & Compatibility

- **Added Dependencies:**
  - `streamlit>=1.35.0`
  - `plotly>=5.20.0`
- **Installation Verification:**
  Install into existing `.venv` without upgrading or modifying `qiskit==1.4.2`, `qiskit-machine-learning==0.8.4`, or `qiskit-aer==0.16.4`.
  Verify with `pytest tests/` after installation to guarantee 100% regression freedom.

---

## 5. Long-Running Experiment Strategy

Evolutionary searches evaluate several quantum circuits (each requiring 25–35 classical optimizer iterations).
- **Execution Mechanism:** Python `threading.Thread` executing `EvolutionarySearchEngine` inside `search_service.py`.
- **Progress Tracking:** Thread-safe state container storing:
  `{status: "idle"|"running"|"completed"|"error", current_gen: int, total_gen: int, best_fitness: float, history: list, error: str|None}`.
- **Safety:** The thread evaluates only on training and validation splits; test evaluation is isolated.
- **Cancellation:** Cooperative cancellation flag checked between generation iterations.
- **No Page Freezing:** Streamlit uses `st.empty()` or `st.rerun(scope="fragment")` or auto-refresh polling while running, allowing navigation to other pages without aborting the search.

---

## 6. Implementation Milestones

1. **Stage C — UI Foundation:**
   - Install `streamlit` and `plotly` in `.venv`.
   - Verify existing 92 tests pass.
   - Implement `ui/styles/theme.css`, `ui/utils/`, and `ui/services/`.
2. **Stage D — Page Implementation (Pages 1–8):**
   - Page 1: Overview Dashboard with KPI cards and Plotly summary charts.
   - Page 2: Dataset Explorer with image filtering, class distributions, and PCA variance.
   - Page 3: Quantum Circuit Lab with dynamic architecture compilation and Qiskit rendering.
   - Page 4: Evolutionary Architecture Search with configuration controls, background execution, and live curves.
   - Page 5: Noise Analysis with interactive Aer depolarizing/readout simulation and retention metrics.
   - Page 6: Model Benchmarking with 7-model comparison table and resource trade-offs.
   - Page 7: Experiment History with JSON loader, search, filter, and comparison.
   - Page 8: Documentation with interactive markdown viewer of project docs.
3. **Stage E — Integration & Testing:**
   - Write and run `tests/test_ui_services.py`, `tests/test_ui_validation.py`, `tests/test_ui_integration.py`.
   - Re-run full test suite (`pytest tests/`).
4. **Stage F — Validation & Documentation:**
   - Launch application on local port.
   - Create documentation deliverables (`PHASE_13_UI_ARCHITECTURE.md`, `PHASE_13_TEST_REPORT.md`, `PHASE_13_USER_GUIDE.md`, `UI_SETUP.md`, updated `README.md`).

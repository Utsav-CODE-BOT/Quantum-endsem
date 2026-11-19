# Phase 13 Backend Interface Audit

## 1. Executive Summary & Audit Mandate

This document provides a comprehensive, evidence-based audit of all existing backend modules, classes, functions, configurations, artifacts, and saved experiment results in the **Self-Adaptive Quantum Neural Networks with Resource-Aware Architecture Optimization** repository.

The goal of this audit is to map every capability of the research codebase to the upcoming Streamlit dashboard (**Quantum Architecture Optimizer**) without modifying core quantum algorithms or fabricating non-existent APIs.

---

## 2. Environment & Dependency Audit

- **Root Directory:** `C:\Users\ud127\.gemini\antigravity\scratch\quantum-architecture-optimizer`
- **Active Python Environment:** Dedicated virtual environment at `.venv` (Python 3.11.9)
- **Key Installed Packages:**
  - `qiskit==1.4.2` (Verified pinned version)
  - `qiskit-aer==0.16.4` (Verified pinned version)
  - `qiskit-machine-learning==0.8.4` (Verified pinned version)
  - `qiskit-algorithms==0.4.0` (Verified pinned version)
  - `scikit-learn==1.9.1` (PCA, SVM, MLP baselines)
  - `scipy==1.15.3` (Optimizer math)
  - `torch==2.15.0+cpu` & `torchvision==0.29.2+cpu` (Fashion-MNIST loader)
  - `numpy==2.4.6`
  - `matplotlib==3.11.2`
  - `pytest==9.2.0` (All 92 existing tests passing)
- **UI Dependency Requirements:**
  - `streamlit`: Not currently installed in `.venv`. Must be installed with compatible version (e.g., `streamlit>=1.35.0`).
  - `plotly`: Not currently installed in `.venv`. Must be installed for interactive charts (`plotly>=5.20.0`).

---

## 3. Comprehensive Backend Interface Inventory

| Module | Function / Class | Inputs | Outputs | Runtime | Side Effects | UI Integration Strategy |
|:---|:---|:---|:---|:---:|:---|:---|
| `src.data.dataset_loader` | `FashionMNISTLoader(raw_data_dir, selected_classes)` | `raw_data_dir: Path\|None`, `selected_classes: Sequence[int]=(0, 9)` | Instance of `FashionMNISTLoader` | Instant (<5ms) | Creates raw dir if missing | Instantiate once; cache in UI dataset service. |
| `src.data.dataset_loader` | `FashionMNISTLoader.load(split="train"\|"test")` | `split: str` | `RawDataset(images, labels, original_labels, label_mapping, class_names)` | ~0.2s–1.0s (cached on disk) | Reads local gzip files | Read-only; safe to call repeatedly with `@st.cache_data`. |
| `src.data.dataset_splitter` | `DatasetSplitter(train_size, val_size, test_size, random_seed)` | Partition sizes & seed | Instance of `DatasetSplitter` | Instant (<1ms) | None | Splits raw data for offline pipelines. |
| `src.preprocessing.preprocessing_pipeline` | `PreprocessingPipeline.load(path)` | `path: str\|Path` | Fitted `PreprocessingPipeline` instance | <50ms | Reads pickle file | Loads fitted PCA components & MinMax scaler parameters. |
| `src.preprocessing.preprocessing_pipeline` | `PreprocessingPipeline.transform(X_raw)` | `X_raw: np.ndarray` (N, 28, 28) | `np.ndarray` (N, 4) in $[0, \pi]$ | <10ms for 100 images | None | Real-time single image or batch preprocessing in Dataset Explorer. |
| `src.quantum.architecture_schema` | `ArchitectureConfig` | `num_qubits, num_layers, rotation_gates, entanglement_type, entanglement_freq, feature_rotation, measurement_strategy` | Immutable frozen dataclass with `.arch_id` property | Instant (<1ms) | None | Core data model for candidate quantum circuits. |
| `src.quantum.ansatz_builder` | `AnsatzBuilder(num_qubits, num_layers, rotation_gates, entanglement_type, ...)` | Architectural parameters | Instance with `.circuit: QuantumCircuit` | <5ms | None | Generates variational ansatz quantum circuit. |
| `src.quantum.circuit_generator` | `VQCCircuitGenerator.from_specs(...)` | Specs matching `ArchitectureConfig` | `VQCCircuitGenerator` with `.full_circuit: QuantumCircuit`, `.metadata: CircuitMetadata` | ~5–15ms | None | Generates combined feature map + ansatz and calculates decomposed depth, gates, and parameters. |
| `src.optimization.architecture_generator` | `ArchitectureGenerator.compile_circuit(config)` | `config: ArchitectureConfig` | `tuple[VQCCircuitGenerator, CircuitMetadata]` | ~10–20ms | None | Primary circuit compilation entrypoint for Circuit Lab. |
| `src.optimization.architecture_space` | `ArchitectureSpace()` | None (uses defaults or config) | `ArchitectureSpace` instance with bounds and `.validate(config)` | Instant (<1ms) | Raises `ArchitectureValidationError` on invalid gene | Validates user-selected parameters in Circuit Lab and Search pages. |
| `src.quantum.vqc_model` | `ConfigurableVQC(...)` | `num_qubits, num_layers, rotation_gates, entanglement_type, maxiter, random_seed, ...` | `ConfigurableVQC` instance | Instant (<10ms init) | None | Model class wrapping Qiskit ML `VQC`. |
| `src.quantum.vqc_model` | `ConfigurableVQC.fit(X, y)` | `X: (N, 4), y: (N,)` | `self` | ~15–30s on 200 samples | Trains classical optimizer (COBYLA) | Long-running operation; run in background thread for live searches. |
| `src.quantum.vqc_model` | `ConfigurableVQC.predict(X)` / `predict_proba(X)` | `X: (N, 4)` | `preds: (N,)`, `probas: (N, 2)` | ~10–30ms per 20 samples | None | Inference is independent of training once fitted. |
| `src.quantum.vqc_model` | `ConfigurableVQC.load_checkpoint(path)` | `path: Path` | `dict` containing weights, metadata, training history | <50ms | Reads pickle file | Allows loading pretrained baseline without rerunning optimization. |
| `src.evaluation.evaluator` | `ArchitectureEvaluator(X_train, y_train, X_val, y_val, ...)` | Preprocessed data arrays | `ArchitectureEvaluator` with `.cache` and `.evaluate(config)` | <10ms init | Tracks evaluations and cache hits | Engine used by evolutionary search. |
| `src.evaluation.evaluator` | `ArchitectureEvaluator.evaluate(config)` | `config: ArchitectureConfig` | `CandidateEvaluationResult` | ~7–25s per candidate (or <1ms if cached) | Caches result in `evaluator.cache` | Evaluates a single circuit configuration. |
| `src.evaluation.fitness` | `FitnessCalculator(w_performance, w_complexity, w_robustness, ...)` | Objective weights | `FitnessCalculator` with `.evaluate(...) -> FitnessBreakdown` | Instant (<1ms) | None | Computes multi-objective fitness. |
| `src.optimization.evolutionary_search` | `EvolutionarySearchEngine(evaluator, pop_size, max_generations, ...)` | Evaluator, population size, generations | `EvolutionarySearchEngine` instance | Instant (<10ms init) | None | Evolutionary search manager. |
| `src.optimization.evolutionary_search` | `EvolutionarySearchEngine.run()` | None | `EvolutionarySearchResult` with generation history | ~60–120s for pop=4, gen=2 | Evaluates population, generates offspring, logs history | Can be executed via thread worker with generation snapshot monitoring. |
| `src.evaluation.noise_evaluator` | `NoiseModelConfig(name, single_qubit_error, two_qubit_error, measurement_error, shots)` | Noise parameters | `NoiseModelConfig` dataclass | Instant (<1ms) | None | Encapsulates noise parameters. |
| `src.evaluation.noise_evaluator` | `NoiseModelBuilder.build(config)` | `config: NoiseModelConfig` | `qiskit_aer.noise.NoiseModel` | ~5ms | Builds Aer depolarizing & readout error model | Safe to call repeatedly. |
| `src.evaluation.noise_evaluator` | `NoiseEvaluator.evaluate(vqc, X_test, y_test, noise_config)` | Fitted `ConfigurableVQC`, test data, noise config | `NoiseEvaluationReport` | ~2–5s (1024 shots Aer density matrix) | Evaluates ideal and noisy predictions | Real-time interactive noise evaluation in Noise Analysis page. |
| `src.baselines.classical_svm` | `ClassicalSVM(kernel, C, ...)` | Kernel type, C, seed | `ClassicalSVM` with `fit`, `predict`, `evaluate` | ~10–30ms fit | Fits scikit-learn SVC | Classical benchmark baseline. |
| `src.baselines.classical_nn` | `ClassicalMLP(hidden_layer_sizes, ...)` | Layer sizes, max_iter | `ClassicalMLP` with `fit`, `predict`, `evaluate` | ~200–300ms fit | Fits scikit-learn MLPClassifier | Classical benchmark baseline. |
| `src.baselines.fixed_vqc` | `create_fixed_shallow_vqc(...)`, `create_fixed_standard_vqc(...)`, `create_fixed_deep_vqc(...)` | Hyperparameters | `ConfigurableVQC` instances with pre-configured topologies | Instant (<10ms) | None | Quantum benchmark baselines. |
| `src.utils.config_loader` | `load_yaml(path)` / `ConfigManager()` | YAML path | `dict` or `ConfigManager` | <10ms | Reads YAML files in `configs/` | Centralized parameter lookup. |

---

## 4. Persisted Research Artifacts Inventory

The existing experiment files and data artifacts are located at:

1. **Preprocessed Data:**
   - `data/processed/fashion_mnist_quantum_ready.npz`:
     Contains: `X_train` (200, 4), `y_train` (200,), `X_val` (60, 4), `y_val` (60,), `X_test` (100, 4), `y_test` (100,).
     Features are scaled to $[0, \pi]$.
   - `data/metadata/preprocessing_metadata.json`:
     Contains PCA explained variance ratio (`[0.4059, 0.1536, 0.0677, 0.0342]`), cumulative variance (`0.6614`), and mean vectors.
   - `data/metadata/preprocessing_pipeline.pkl`:
     Serialized `PreprocessingPipeline` instance.
   - `data/raw/FashionMNIST/raw/`:
     Full raw binary Fashion-MNIST images and labels.

2. **Trained Checkpoints:**
   - `experiments/checkpoints/baseline_vqc.pkl`:
     Trained baseline VQC weights, circuit metadata (depth 12, 24 params), and training history.

3. **Experiment Results:**
   - `experiments/results/baseline_vqc_results.json`:
     Validation acc: 98.33%, Test acc: 95.00%, Depth: 12, Total gates: 34, Parameters: 24.
   - `experiments/results/best_architecture.json`:
     Best candidate (`arch_6327d7bcb8`): Depth 5, Total gates 20, 0 CX gates, 16 parameters, Validation acc: 91.67%, Test acc: 95.00%, Fitness: 0.6433.
   - `experiments/results/search_history.json`:
     Generation history (Gen 1 best: 0.6150, Gen 2 best: 0.6433, total evaluations: 6, cache hits: 2, runtime: 93.21s).
   - `experiments/results/noise_experiment_results.json`:
     Paired ideal vs noisy evaluations across Low, Medium, High noise tiers for both Baseline and Search-Optimized VQC.
   - `experiments/results/benchmark_comparison.json`:
     7-model comparative benchmark records (Linear SVM, RBF SVM, MLP, Shallow VQC, Standard VQC, Deep VQC, Optimized VQC).
   - `experiments/results/RESEARCH_REPORT.md`:
     Compiled comprehensive research report.

4. **Generated Figures:**
   - `experiments/figures/fig1_search_convergence.png`
   - `experiments/figures/fig2_pareto_tradeoff.png`
   - `experiments/figures/fig3_model_comparison.png`
   - `experiments/figures/fig4_noise_degradation.png`
   - `experiments/figures/fig5_confusion_matrices.png`

---

## 5. Architectural Findings & Behavioral Constraints

1. **Read-Only Loading vs Live Execution:**
   All 8 pages can load and display existing results immediately from persisted JSON/NPZ files without running any quantum simulations. When the researcher clicks "Run Search" or "Run Noise Evaluation", the live backend is invoked.
2. **Computational Expense & Threading:**
   - Training 1 VQC candidate takes ~7–25 seconds.
   - An evolutionary search of population=4 across 2 generations takes ~90 seconds.
   - Streamlit must NOT run this on the main script thread without background execution management, otherwise the UI will freeze and timeout.
   - A thread-based worker with a thread-safe progress container or state object is ideal for local single-user execution without Celery/Redis overhead.
3. **Qiskit Circuit Rendering:**
   `QuantumCircuit.draw(output='mpl')` produces high-quality Matplotlib figures, while `output='text'` produces clean ASCII representations. Both can be rendered natively in Streamlit (`st.pyplot(fig)` and `st.code(text)`).
4. **Data Leakage Safeguards:**
   - Preprocessing transforms test/validation data using the pre-fitted PCA and scaler.
   - Test data is NEVER used in `ArchitectureEvaluator.evaluate()` for evolutionary selection; validation data alone drives fitness.
   - The UI will explicitly label validation vs test metrics to prevent leakage confusion.

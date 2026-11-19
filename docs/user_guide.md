# User Guide & Command-Line Interface Manual

## 1. Quickstart Workflow

Run the entire end-to-end research pipeline using the dedicated command-line scripts:

```powershell
# Activate isolated virtual environment
.\.venv\Scripts\Activate.ps1

# Step 1: Preprocess Fashion-MNIST into quantum-ready features
python scripts/run_preprocessing.py --n-train 200 --n-val 60 --n-test 100 --n-components 4

# Step 2: Train the Baseline VQC
python scripts/run_baseline.py --maxiter 35

# Step 3: Run Evolutionary Architecture Search
python scripts/run_architecture_search.py --pop-size 4 --generations 2 --maxiter 25

# Step 4: Run Quantum Hardware Noise Robustness Experiments
python scripts/run_noise_experiments.py

# Step 5: Run Comprehensive Classical & Quantum Benchmarks
python scripts/run_benchmarks.py

# Step 6: Compile Publication Figures and Final Research Report
python scripts/generate_report.py
```

---

## 2. Configuration System

Global parameters are configured in `configs/default_config.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  dataset_name: "fashion_mnist"
  classes: [0, 9]  # 0: T-shirt, 9: Ankle boot
  train_samples: 200
  val_samples: 60
  test_samples: 100

preprocessing:
  pca_components: 4
  scale_min: 0.0
  scale_max: 3.141592653589793  # pi

quantum:
  num_qubits: 4
  feature_map: "angle"
  rotation_gate: "ry"
  default_ansatz_layers: 2
  default_rotation_gates: ["ry", "rz"]
  default_entanglement: "linear"
  measurement: "parity"

optimization:
  population_size: 4
  generations: 2
  tournament_size: 2
  crossover_prob: 0.70
  mutation_prob: 0.30
  weights:
    performance: 0.50
    complexity: 0.30
    robustness: 0.20
```

---

## 3. Running Unit and Integration Tests

The test suite covers dataset splitting, zero-leakage PCA, ansatz generation, circuit metrics, VQC training, evolutionary operators, noise simulation, and classical baselines:

```powershell
# Run full test suite
pytest tests/ -v

# Run with test coverage
pytest tests/ --cov=src --cov-report=term-missing
```

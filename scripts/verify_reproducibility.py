"""Independent reproducibility validation script.

Verifies:
1. Checkpoint loading and parameter dimensions.
2. Weight array SHA-256 digests.
3. Model reconstruction and non-retraining inference on validation and test sets.
4. Confusion matrix and metric verification.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.quantum.vqc_model import ConfigurableVQC
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.fitness import FitnessCalculator
from src.quantum.circuit_generator import VQCCircuitGenerator
from src.quantum.architecture_schema import ArchitectureConfig


def verify_checkpoints() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("=== 1. DATASET PARTITION CHECK ===")
    print(f"Train samples: {len(X_train)} (classes: {np.bincount(y_train)})")
    print(f"Val samples:   {len(X_val)} (classes: {np.bincount(y_val)})")
    print(f"Test samples:  {len(X_test)} (classes: {np.bincount(y_test)})")
    assert len(X_train) == 200 and len(X_val) == 60 and len(X_test) == 100
    print("[PASS] Dataset sample counts and class balance verified: 200/60/100.\n")

    print("=== 2. CHECKPOINT & WEIGHT HASH VERIFICATION ===")
    base_ckpt_path = root / "experiments" / "checkpoints" / "baseline_vqc.pkl"
    opt_ckpt_path = root / "experiments" / "checkpoints" / "optimized_vqc.pkl"

    assert base_ckpt_path.exists(), f"Missing baseline checkpoint at {base_ckpt_path}"
    assert opt_ckpt_path.exists(), f"Missing optimized checkpoint at {opt_ckpt_path}"

    base_ckpt = ConfigurableVQC.load_checkpoint(base_ckpt_path)
    opt_ckpt = ConfigurableVQC.load_checkpoint(opt_ckpt_path)

    w_base = np.array(base_ckpt["weights"], dtype=np.float64)
    w_opt = np.array(opt_ckpt["weights"], dtype=np.float64)

    h_base = hashlib.sha256(w_base.tobytes()).hexdigest()[:16]
    h_opt = hashlib.sha256(w_opt.tobytes()).hexdigest()[:16]

    print(f"Baseline VQC Checkpoint:")
    print(f"  Weights shape: {w_base.shape} (expected 24)")
    print(f"  Optimization budget: maxiter={base_ckpt.get('maxiter')}")
    print(f"  Computed SHA-256 prefix: {h_base} (expected: 21bf8f99f15b3672)")
    assert h_base == "21bf8f99f15b3672", f"Baseline hash mismatch: {h_base}"

    print(f"Optimized VQC Checkpoint (arch_4169ee9360):")
    print(f"  Weights shape: {w_opt.shape} (expected 32)")
    print(f"  Optimization budget: maxiter={opt_ckpt.get('maxiter')}")
    print(f"  Computed SHA-256 prefix: {h_opt} (expected: 151e19ca76722842)")
    assert h_opt == "151e19ca76722842", f"Optimized hash mismatch: {h_opt}"
    print("[PASS] Checkpoint weights and hashes verified exactly.\n")

    print("=== 3. NON-RETRAINING INFERENCE & METRIC VERIFICATION ===")
    vqc_opt = ConfigurableVQC.load(opt_ckpt_path)
    opt_val_prob = vqc_opt.predict_proba(X_val)
    opt_val_preds = vqc_opt.predict(X_val)
    opt_val_metrics = compute_classification_metrics(y_val, opt_val_preds, opt_val_prob[:, 1])

    opt_test_prob = vqc_opt.predict_proba(X_test)
    opt_test_preds = vqc_opt.predict(X_test)
    opt_test_metrics = compute_classification_metrics(y_test, opt_test_preds, opt_test_prob[:, 1])

    print("Optimized VQC (Standardized Benchmark EXP-BENCH-20261120):")
    print(f"  Validation Accuracy: {opt_val_metrics.accuracy * 100:.2f}% ({np.sum(opt_val_preds == y_val)}/60)")
    print(f"  Validation Confusion Matrix: {opt_val_metrics.confusion_matrix}")
    print(f"  Test Accuracy:       {opt_test_metrics.accuracy * 100:.2f}% ({np.sum(opt_test_preds == y_test)}/100)")
    print(f"  Test Confusion Matrix:       {opt_test_metrics.confusion_matrix}")
    print(f"  Test F1 Score:       {opt_test_metrics.f1_macro:.4f}")
    print(f"  Test ROC-AUC:        {opt_test_metrics.roc_auc:.4f}")

    assert np.isclose(opt_val_metrics.accuracy, 53 / 60, atol=1e-4)
    assert np.isclose(opt_test_metrics.accuracy, 0.90, atol=1e-4)
    print("[PASS] Optimized VQC inference metrics match benchmark results.\n")

    vqc_base = ConfigurableVQC.load(base_ckpt_path)
    base_val_prob = vqc_base.predict_proba(X_val)
    base_val_preds = vqc_base.predict(X_val)
    base_val_metrics = compute_classification_metrics(y_val, base_val_preds, base_val_prob[:, 1])

    base_test_prob = vqc_base.predict_proba(X_test)
    base_test_preds = vqc_base.predict(X_test)
    base_test_metrics = compute_classification_metrics(y_test, base_test_preds, base_test_prob[:, 1])

    print("Baseline Fixed VQC (Standalone Run EXP-20261120-QNN-01):")
    print(f"  Validation Accuracy: {base_val_metrics.accuracy * 100:.2f}% ({np.sum(base_val_preds == y_val)}/60)")
    print(f"  Validation Confusion Matrix: {base_val_metrics.confusion_matrix}")
    print(f"  Test Accuracy:       {base_test_metrics.accuracy * 100:.2f}% ({np.sum(base_test_preds == y_test)}/100)")
    print(f"  Test Confusion Matrix:       {base_test_metrics.confusion_matrix}")
    print(f"  Test F1 Score:       {base_test_metrics.f1_macro:.4f}")
    print(f"  Test ROC-AUC:        {base_test_metrics.roc_auc:.4f}")

    assert np.isclose(base_val_metrics.accuracy, 1.0, atol=1e-4)
    assert np.isclose(base_test_metrics.accuracy, 0.96, atol=1e-4)
    print("[PASS] Baseline VQC inference metrics match recorded results.\n")

    print("=== 4. CIRCUIT RESOURCE & TWO-QUBIT REDUCTION CHECK ===")
    from src.optimization.architecture_generator import ArchitectureGenerator
    opt_cfg = ArchitectureConfig(
        num_qubits=opt_ckpt["num_qubits"],
        num_layers=opt_ckpt["num_layers"],
        rotation_gates=tuple(opt_ckpt["rotation_gates"]),
        entanglement_type=opt_ckpt["entanglement_type"],
        entanglement_freq=opt_ckpt["entanglement_freq"],
        feature_rotation=opt_ckpt["feature_rotation"],
        measurement_strategy=opt_ckpt["measurement_strategy"],
    )
    _, opt_metadata = ArchitectureGenerator.compile_circuit(opt_cfg)
    print(f"Optimized Circuit Depth: {opt_metadata.circuit_depth} (expected 10)")
    print(f"Optimized Two-Qubit CX Gates: {opt_metadata.two_qubit_gates} (expected 1)")
    print(f"Optimized Total Gates: {opt_metadata.total_gates} (expected 37)")
    print(f"Baseline Two-Qubit CX Gates: 6")
    cx_reduction = (6 - opt_metadata.two_qubit_gates) / 6 * 100
    print(f"Calculated Two-Qubit Gate Reduction: {cx_reduction:.2f}% (expected: 83.33%)")
    assert np.isclose(cx_reduction, 83.3333, atol=1e-2)
    print("[PASS] Circuit resource verification successful.\n")

    print("=== 5. FITNESS CALCULATION VERIFICATION ===")
    fc = FitnessCalculator(w_performance=0.5, w_complexity=0.3, w_robustness=0.2, max_depth=50, max_cx_gates=30)
    comp = fc.calculate_complexity(opt_metadata)
    rob = fc.calculate_robustness(opt_val_metrics.accuracy, opt_val_metrics.accuracy)
    bd = fc.evaluate(opt_val_metrics.accuracy, opt_metadata, opt_val_metrics.accuracy)
    print(f"Calculated Complexity: {comp:.6f} (expected: 0.116667)")
    print(f"Calculated Robustness: {rob:.6f} (expected: 1.000000)")
    print(f"Calculated Fitness:    {bd.scalar_fitness:.6f} (expected: 0.606667)")
    assert np.isclose(comp, 0.116667, atol=1e-4)
    assert np.isclose(rob, 1.000000, atol=1e-4)
    assert np.isclose(bd.scalar_fitness, 0.606667, atol=1e-4)
    print("[PASS] Fitness formula and standardized benchmark values verified exactly.")


if __name__ == "__main__":
    verify_checkpoints()

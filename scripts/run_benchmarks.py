"""Comprehensive benchmark runner comparing classical baselines, fixed VQCs, and the search-optimized VQC.

Models evaluated:
1. Classical Linear SVM
2. Classical RBF SVM
3. Classical MLP (comparable ~30 parameter capacity)
4. Fixed Shallow VQC (1 layer)
5. Fixed Standard Baseline VQC (2 layers)
6. Fixed Deep VQC (4 layers)
7. Evolutionary Search-Optimized VQC (compact Pareto candidate)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.baselines.classical_nn import ClassicalMLP
from src.baselines.classical_svm import ClassicalSVM
from src.baselines.fixed_vqc import (
    create_fixed_deep_vqc,
    create_fixed_shallow_vqc,
    create_fixed_standard_vqc,
)
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.config_loader import get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_inference_latency(model: Any, X: np.ndarray, num_runs: int = 5) -> float:
    """Measure average inference latency per sample in milliseconds."""
    t_start = time.perf_counter()
    for _ in range(num_runs):
        _ = model.predict(X)
    total_time = time.perf_counter() - t_start
    return (total_time / (num_runs * len(X))) * 1000.0


def main() -> None:
    logger.info("=" * 80)
    logger.info("PHASE 9: COMPREHENSIVE CLASSICAL & QUANTUM BENCHMARKING")
    logger.info("=" * 80)

    proj_root = get_project_root()
    data_path = proj_root / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run preprocessing first.")

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info("Dataset: Train=%d, Val=%d, Test=%d, Features=%d", len(X_train), len(X_val), len(X_test), X_train.shape[1])

    # 1. Instantiate Search-Optimized VQC
    best_arch_file = proj_root / "experiments" / "results" / "best_architecture.json"
    if best_arch_file.exists():
        with open(best_arch_file, "r") as f:
            best_arch_data = json.load(f)
        best_cfg = best_arch_data["config"]
        optimized_vqc = ConfigurableVQC(
            num_qubits=int(best_cfg["num_qubits"]),
            num_layers=int(best_cfg["num_layers"]),
            rotation_gates=list(best_cfg["rotation_gates"]),
            entanglement_type=str(best_cfg["entanglement_type"]),
            entanglement_freq=int(best_cfg.get("entanglement_freq", 1)),
            maxiter=35,
            random_seed=42,
        )
        opt_name = f"Optimized VQC ({best_arch_data.get('arch_id', 'search')})"
    else:
        optimized_vqc = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["rx", "ry"],
            entanglement_type="circular",
            entanglement_freq=2,
            maxiter=35,
            random_seed=42,
        )
        opt_name = "Optimized VQC (compact)"

    # Define model suite
    models: list[tuple[str, str, Any, dict[str, Any]]] = [
        ("Linear SVM", "Classical", ClassicalSVM(kernel="linear", C=1.0, random_seed=42), {}),
        ("RBF SVM", "Classical", ClassicalSVM(kernel="rbf", C=1.0, random_seed=42), {}),
        ("Classical MLP", "Classical", ClassicalMLP(hidden_layer_sizes=(4,), max_iter=300, random_seed=42), {}),
        ("Fixed Shallow VQC", "Quantum Baseline", create_fixed_shallow_vqc(maxiter=35, random_seed=42), {}),
        ("Fixed Standard VQC", "Quantum Baseline", create_fixed_standard_vqc(maxiter=35, random_seed=42), {}),
        ("Fixed Deep VQC", "Quantum Baseline", create_fixed_deep_vqc(maxiter=35, random_seed=42), {}),
        (opt_name, "Quantum Search", optimized_vqc, {}),
    ]

    benchmark_records: list[dict[str, Any]] = []

    logger.info("\n" + "=" * 90)
    logger.info(f"{'Model Name':<28} | {'Category':<16} | {'Params':<6} | {'Depth':<5} | {'Val Acc':<8} | {'Test Acc':<8} | {'Train Time':<10}")
    logger.info("=" * 90)

    for name, category, model, extra in models:
        # Fit model
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Evaluate on validation and test
        val_report = model.evaluate(X_val, y_val)
        test_report = model.evaluate(X_test, y_test)

        # Latency
        latency_ms = evaluate_inference_latency(model, X_test[:20], num_runs=3)

        # Hardware metrics
        if hasattr(model, "metadata"):
            depth = model.metadata.circuit_depth
            total_gates = model.metadata.total_gates
            two_qubit_gates = model.metadata.two_qubit_gates
            n_params = model.metadata.num_trainable_parameters
        else:
            depth = None
            total_gates = None
            two_qubit_gates = None
            n_params = getattr(model, "num_parameters", 0)

        record = {
            "model_name": name,
            "category": category,
            "num_parameters": n_params,
            "circuit_depth": depth,
            "total_gates": total_gates,
            "two_qubit_gates": two_qubit_gates,
            "val_accuracy": round(val_report.accuracy, 4),
            "val_f1": round(val_report.f1_macro, 4),
            "test_accuracy": round(test_report.accuracy, 4),
            "test_precision": round(test_report.precision_macro, 4),
            "test_recall": round(test_report.recall_macro, 4),
            "test_f1": round(test_report.f1_macro, 4),
            "test_roc_auc": round(test_report.roc_auc if test_report.roc_auc is not None else 0.0, 4),
            "training_time_seconds": round(train_time, 2),
            "latency_ms_per_sample": round(latency_ms, 3),
        }
        benchmark_records.append(record)

        depth_str = str(depth) if depth is not None else "-"
        logger.info(
            f"{name:<28} | {category:<16} | {n_params:<6} | {depth_str:<5} | "
            f"{val_report.accuracy * 100:>6.2f}% | {test_report.accuracy * 100:>6.2f}% | {train_time:>8.2f}s"
        )

    logger.info("=" * 90)

    # Save to JSON
    out_dir = proj_root / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_comparison.json"

    with open(out_file, "w") as f:
        json.dump(
            {
                "task": "Fashion-MNIST Binary Classification (0: T-shirt vs 9: Ankle Boot)",
                "num_features": 4,
                "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
                "results": benchmark_records,
            },
            f,
            indent=2,
        )

    logger.info("\nSaved benchmark comparison results to: %s", out_file)


if __name__ == "__main__":
    main()

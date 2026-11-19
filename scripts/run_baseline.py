#!/usr/bin/env python
"""Train and evaluate the baseline Variational Quantum Classifier (Phase 4).

Usage:
    python scripts/run_baseline.py [--maxiter 50] [--n-samples 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.classification_metrics import compute_classification_metrics
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.config_loader import ConfigManager, get_project_root
from src.utils.logging_config import get_logger, setup_logging
from src.utils.system_monitor import Timer, get_resource_snapshot

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline VQC")
    parser.add_argument("--maxiter", type=int, default=None, help="Optimizer iterations")
    parser.add_argument("--n-train-samples", type=int, default=None, help="Max training samples to use")
    args = parser.parse_args()

    setup_logging(level="INFO", log_dir="experiments/logs", log_file="run_baseline.log")
    cfg = ConfigManager()

    # 1. Load preprocessed quantum-ready dataset
    proj_root = get_project_root()
    data_path = proj_root / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
    if not data_path.exists():
        logger.error("Preprocessed data not found at %s. Please run scripts/run_preprocessing.py first.", data_path)
        sys.exit(1)

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if args.n_train_samples is not None and args.n_train_samples < len(X_train):
        X_train = X_train[: args.n_train_samples]
        y_train = y_train[: args.n_train_samples]

    maxiter = args.maxiter or cfg.get("training", "training.optimizer.maxiter", 50)
    optimizer_name = cfg.get("training", "training.optimizer.name", "cobyla")
    seed = cfg.get("training", "training.parameter_init_seed", 42)

    logger.info("Initializing Baseline VQC: 4 qubits, 2 layers, linear entanglement...")
    vqc = ConfigurableVQC(
        num_qubits=4,
        num_layers=2,
        rotation_gates=("ry", "rz"),
        entanglement_type="linear",
        entanglement_freq=1,
        feature_rotation="ry",
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        random_seed=seed,
        measurement_strategy="parity",
    )

    mem_before = get_resource_snapshot()

    # 2. Train baseline VQC with execution timer
    print("\n" + "=" * 60)
    print("BASELINE VQC TRAINING & BENCHMARKING (PHASE 4)")
    print("=" * 60)
    print(f"Dataset:            Fashion-MNIST (T-shirt vs Ankle boot)")
    print(f"Training Samples:   {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    print(f"Test Samples:       {len(X_test)}")
    print(f"Optimizer:          {optimizer_name.upper()} (maxiter={maxiter})")
    print(f"Circuit Depth:      {vqc.metadata.circuit_depth}")
    print(f"Total Gates:        {vqc.metadata.total_gates} (CX: {vqc.metadata.two_qubit_gates})")
    print(f"Trainable Params:   {vqc.metadata.num_trainable_parameters}")
    print("-" * 60)
    print("Training in progress (evaluating quantum cost function)...")

    with Timer("Baseline VQC Training", log=True) as t:
        vqc.fit(X_train, y_train)

    train_time = t.elapsed
    mem_after = get_resource_snapshot()

    # 3. Validation evaluation
    val_preds = vqc.predict(X_val)
    val_probs = vqc.predict_proba(X_val)
    val_metrics = compute_classification_metrics(y_val, val_preds, val_probs)

    # 4. Test evaluation
    test_preds = vqc.predict(X_test)
    test_probs = vqc.predict_proba(X_test)
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)

    # Calculate empirical throughput metrics for budget planning (Technical Clarification #3)
    n_evals = vqc.training_history.total_evaluations
    sec_per_eval = train_time / max(n_evals, 1)
    sec_per_sample_eval = train_time / (max(n_evals, 1) * len(X_train))

    benchmark_stats = {
        "training_time_seconds": round(train_time, 2),
        "total_evaluations": n_evals,
        "seconds_per_evaluation": round(sec_per_eval, 4),
        "seconds_per_sample_evaluation": round(sec_per_sample_eval, 6),
        "circuit_metadata": vqc.metadata.to_dict(),
        "validation_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "memory_ram_used_gb": mem_after.ram_used_gb,
    }

    # Save checkpoint and results
    checkpoints_dir = proj_root / "experiments" / "checkpoints"
    results_dir = proj_root / "experiments" / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    vqc.save(checkpoints_dir / "baseline_vqc.pkl")
    with open(results_dir / "baseline_vqc_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE VQC EVALUATION & RESOURCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Training Time:           {train_time:.2f} seconds")
    print(f"Total Cost Evaluations:  {n_evals}")
    print(f"Empirical Evaluation:    {sec_per_eval:.4f} sec/iteration ({sec_per_sample_eval * 1e3:.2f} ms/sample-eval)")
    print(f"Final Training Loss:     {vqc.training_history.best_loss:.4f}")
    print("-" * 60)
    print(f"Validation Accuracy:     {val_metrics.accuracy * 100:.2f}%")
    print(f"Validation F1 (Macro):   {val_metrics.f1_macro:.4f}")
    if val_metrics.roc_auc is not None:
        print(f"Validation ROC-AUC:      {val_metrics.roc_auc:.4f}")
    print(f"Validation Confusion Matrix:\n  {val_metrics.confusion_matrix}")
    print("-" * 60)
    print(f"Test Accuracy:           {test_metrics.accuracy * 100:.2f}%")
    print(f"Test F1 (Macro):         {test_metrics.f1_macro:.4f}")
    if test_metrics.roc_auc is not None:
        print(f"Test ROC-AUC:            {test_metrics.roc_auc:.4f}")
    print(f"Test Confusion Matrix:\n  {test_metrics.confusion_matrix}")
    print("-" * 60)
    print(f"Model Checkpoint:        {checkpoints_dir / 'baseline_vqc.pkl'}")
    print(f"Results File:            {results_dir / 'baseline_vqc_results.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

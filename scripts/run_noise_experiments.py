"""Run comprehensive noise robustness experiments comparing Baseline VQC and Search-Optimized VQC.

Evaluates performance across realistic NISQ noise regimes:
- Ideal (statevector simulation)
- Low Noise (1q=0.05%, 2q=0.5%, readout=0.5%)
- Medium Noise (1q=0.1%, 2q=1.0%, readout=1.5%)
- High Noise (1q=0.5%, 2q=3.0%, readout=3.0%)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.noise_evaluator import (
    NoiseEvaluator,
    NoiseModelConfig,
)
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.config_loader import get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("PHASE 8: QUANTUM HARDWARE NOISE ROBUSTNESS BENCHMARKING")
    logger.info("=" * 70)

    proj_root = get_project_root()

    # 1. Load preprocessed Fashion-MNIST quantum-ready data
    data_path = proj_root / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run preprocessing first.")

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info("Loaded dataset: Train=%d, Test=%d, Features=%d", len(X_train), len(X_test), X_train.shape[1])

    # 2. Setup Baseline VQC model
    logger.info("\n--- Initializing & Training Baseline VQC ---")
    baseline_vqc = ConfigurableVQC(
        num_qubits=4,
        num_layers=2,
        rotation_gates=["ry", "rz"],
        entanglement_type="linear",
        maxiter=35,
        random_seed=42,
    )
    setattr(baseline_vqc, "arch_id", "baseline_vqc_depth12")
    t0 = time.perf_counter()
    baseline_vqc.fit(X_train, y_train)
    t_base = time.perf_counter() - t0
    logger.info("Baseline VQC trained in %.2fs (Best loss: %.4f)", t_base, baseline_vqc.training_history.best_loss or 0.0)

    # 3. Setup Search-Optimized VQC model
    logger.info("\n--- Initializing & Training Search-Optimized VQC ---")
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
        opt_id = best_arch_data.get("arch_id", "optimized_vqc")
    else:
        logger.warning("best_architecture.json not found; using fallback optimized configuration.")
        optimized_vqc = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["rx", "ry"],
            entanglement_type="circular",
            entanglement_freq=2,
            maxiter=35,
            random_seed=42,
        )
        opt_id = "arch_optimized_compact"

    setattr(optimized_vqc, "arch_id", opt_id)
    t0 = time.perf_counter()
    optimized_vqc.fit(X_train, y_train)
    t_opt = time.perf_counter() - t0
    logger.info("Optimized VQC trained in %.2fs (Best loss: %.4f)", t_opt, optimized_vqc.training_history.best_loss or 0.0)

    # 4. Define Noise Tiers
    noise_tiers = [
        NoiseModelConfig(
            name="Low_Noise",
            single_qubit_error=0.0005,
            two_qubit_error=0.005,
            measurement_error=0.005,
            shots=1024,
        ),
        NoiseModelConfig(
            name="Medium_Noise",
            single_qubit_error=0.001,
            two_qubit_error=0.010,
            measurement_error=0.015,
            shots=1024,
        ),
        NoiseModelConfig(
            name="High_Noise",
            single_qubit_error=0.005,
            two_qubit_error=0.030,
            measurement_error=0.030,
            shots=1024,
        ),
    ]

    # 5. Run Noise Benchmarks
    evaluator = NoiseEvaluator(seed=42)
    results: dict[str, list[dict]] = {
        "baseline": [],
        "optimized": [],
    }

    logger.info("\n" + "=" * 70)
    logger.info(f"{'Tier':<14} | {'Model':<12} | {'Ideal Acc':<10} | {'Noisy Acc':<10} | {'Drop':<8} | {'Retention':<10}")
    logger.info("=" * 70)

    for tier in noise_tiers:
        # Evaluate Baseline
        b_report = evaluator.evaluate(baseline_vqc, X_test, y_test, tier)
        results["baseline"].append(b_report.to_dict())
        logger.info(
            f"{tier.name:<14} | {'Baseline':<12} | {b_report.ideal_metrics.accuracy * 100:>8.2f}% | "
            f"{b_report.noisy_metrics.accuracy * 100:>8.2f}% | {b_report.accuracy_drop * 100:>6.2f}% | "
            f"{b_report.relative_accuracy_retention * 100:>8.2f}%"
        )

        # Evaluate Optimized
        o_report = evaluator.evaluate(optimized_vqc, X_test, y_test, tier)
        results["optimized"].append(o_report.to_dict())
        logger.info(
            f"{tier.name:<14} | {'Optimized':<12} | {o_report.ideal_metrics.accuracy * 100:>8.2f}% | "
            f"{o_report.noisy_metrics.accuracy * 100:>8.2f}% | {o_report.accuracy_drop * 100:>6.2f}% | "
            f"{o_report.relative_accuracy_retention * 100:>8.2f}%"
        )
        logger.info("-" * 70)

    # 6. Save results to experiments/results/noise_experiment_results.json
    out_dir = proj_root / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "noise_experiment_results.json"

    metadata = {
        "dataset": "Fashion-MNIST (0: T-shirt vs 9: Ankle Boot)",
        "features": 4,
        "test_samples": len(X_test),
        "baseline_architecture": {
            "num_layers": baseline_vqc.num_layers,
            "depth": baseline_vqc.metadata.circuit_depth,
            "total_gates": baseline_vqc.metadata.total_gates,
            "num_parameters": baseline_vqc.metadata.num_trainable_parameters,
        },
        "optimized_architecture": {
            "num_layers": optimized_vqc.num_layers,
            "depth": optimized_vqc.metadata.circuit_depth,
            "total_gates": optimized_vqc.metadata.total_gates,
            "num_parameters": optimized_vqc.metadata.num_trainable_parameters,
        },
        "benchmarks": results,
    }

    with open(out_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nSaved noise experiment results to: %s", out_file)


if __name__ == "__main__":
    main()

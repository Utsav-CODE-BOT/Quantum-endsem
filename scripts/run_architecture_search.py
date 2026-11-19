#!/usr/bin/env python
"""Execute Evolutionary Architecture Search on Fashion-MNIST (Phase 7).

Usage:
    python scripts/run_architecture_search.py [--pop-size 6] [--generations 3] [--maxiter 25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.evaluator import ArchitectureEvaluator
from src.evaluation.fitness import FitnessCalculator
from src.optimization.evolutionary_search import EvolutionarySearchEngine
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.config_loader import ConfigManager, get_project_root
from src.utils.logging_config import get_logger, setup_logging
from src.utils.seed_manager import SeedManager

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evolutionary Architecture Search")
    parser.add_argument("--pop-size", type=int, default=None, help="Population size")
    parser.add_argument("--generations", type=int, default=None, help="Max generations")
    parser.add_argument("--maxiter", type=int, default=None, help="VQC optimizer iterations")
    parser.add_argument("--n-train-samples", type=int, default=None, help="Number of training samples")
    args = parser.parse_args()

    setup_logging(level="INFO", log_dir="experiments/logs", log_file="run_architecture_search.log")
    cfg = ConfigManager()

    # 1. Load preprocessed quantum-ready dataset
    proj_root = get_project_root()
    data_path = proj_root / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
    if not data_path.exists():
        logger.error("Preprocessed data not found. Please run scripts/run_preprocessing.py first.")
        sys.exit(1)

    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    if args.n_train_samples is not None and args.n_train_samples < len(X_train):
        X_train = X_train[: args.n_train_samples]
        y_train = y_train[: args.n_train_samples]

    pop_size = args.pop_size or cfg.get("evolutionary_search", "evolutionary_search.population_size", 6)
    generations = args.generations or cfg.get("evolutionary_search", "evolutionary_search.max_generations", 3)
    maxiter = args.maxiter or 25
    elite_count = cfg.get("evolutionary_search", "evolutionary_search.elite_count", 2)
    seed = cfg.get("evolutionary_search", "evolutionary_search.search_seed", 42)

    fitness_cfg = cfg.get("evolutionary_search", "evolutionary_search.fitness", {})
    fitness_calc = FitnessCalculator(
        w_performance=fitness_cfg.get("w_performance", 0.5),
        w_complexity=fitness_cfg.get("w_complexity", 0.3),
        w_robustness=fitness_cfg.get("w_noise_robustness", 0.2),
    )

    seed_mgr = SeedManager(global_seed=seed)

    print("\n" + "=" * 65)
    print("EVOLUTIONARY ARCHITECTURE SEARCH (PHASE 7)")
    print("=" * 65)
    print(f"Population Size:       {pop_size}")
    print(f"Generations:           {generations}")
    print(f"Elites Preserved:      {elite_count}")
    print(f"Candidate MaxIter:     {maxiter}")
    print(f"Training Samples:      {len(X_train)} (Features: {X_train.shape[1]})")
    print(f"Validation Samples:    {len(X_val)}")
    print(f"Objective Weights:     Perf={fitness_calc.w_perf}, Comp={fitness_calc.w_comp}, Rob={fitness_calc.w_rob}")
    print("-" * 65)
    print("Beginning evolutionary search iterations...")

    # 2. Setup Evaluator and Search Engine
    evaluator = ArchitectureEvaluator(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fitness_calculator=fitness_calc,
        maxiter=maxiter,
        training_seed=seed_mgr.derive_seed("vqc_evaluator"),
        enable_cache=True,
    )

    engine = EvolutionarySearchEngine(
        evaluator=evaluator,
        population_size=pop_size,
        max_generations=generations,
        elite_count=elite_count,
        seed_manager=seed_mgr,
    )

    # 3. Execute search
    search_result = engine.run()

    # 4. Final held-out test evaluation on the best selected architecture ONLY
    best_cand = search_result.best_candidate
    best_config = best_cand.config

    logger.info("Evaluating selected best architecture %s on held-out test set...", best_cand.arch_id)
    final_vqc = ConfigurableVQC(
        num_qubits=best_config.num_qubits,
        num_layers=best_config.num_layers,
        rotation_gates=best_config.rotation_gates,
        entanglement_type=best_config.entanglement_type,
        entanglement_freq=best_config.entanglement_freq,
        maxiter=maxiter,
        random_seed=seed,
    )
    final_vqc.fit(X_train, y_train)
    test_preds = final_vqc.predict(X_test)
    test_probs = final_vqc.predict_proba(X_test)
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    best_cand.test_metrics = test_metrics

    # 5. Save results
    results_dir = proj_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / "search_history.json"
    search_result.save(history_path)

    best_arch_path = results_dir / "best_architecture.json"
    with open(best_arch_path, "w", encoding="utf-8") as f:
        json.dump(best_cand.to_dict(), f, indent=2)

    # Print summary
    print("\n" + "=" * 65)
    print("EVOLUTIONARY ARCHITECTURE SEARCH COMPLETED")
    print("=" * 65)
    print(f"Total Search Runtime:  {search_result.total_runtime_seconds:.2f} seconds")
    print(f"Total Model Evals:     {search_result.total_evaluations} (Unique: {search_result.unique_architectures_evaluated}, Cache hits: {search_result.cache_hits})")
    print(f"Termination Reason:    {search_result.stop_reason}")
    print("-" * 65)
    print(f"BEST ARCHITECTURE:     {best_cand.arch_id}")
    print(f"  Ansatz Depth:        {best_config.num_layers} layers")
    print(f"  Rotation Gates:      {list(best_config.rotation_gates)}")
    print(f"  Entanglement:        '{best_config.entanglement_type}' (freq={best_config.entanglement_freq})")
    print(f"  Circuit Depth:       {best_cand.circuit_metadata.circuit_depth}")
    print(f"  Total Gates:         {best_cand.circuit_metadata.total_gates} (CX: {best_cand.circuit_metadata.two_qubit_gates})")
    print(f"  Trainable Params:    {best_cand.circuit_metadata.num_trainable_parameters}")
    print("-" * 65)
    print(f"VALIDATION PERFORMANCE (Selection Criterion):")
    print(f"  Fitness Score:       {best_cand.fitness:.4f}")
    print(f"  Accuracy:            {best_cand.val_metrics.accuracy * 100:.2f}%")
    print(f"  F1 Score (Macro):    {best_cand.val_metrics.f1_macro:.4f}")
    if best_cand.val_metrics.roc_auc is not None:
        print(f"  ROC-AUC:             {best_cand.val_metrics.roc_auc:.4f}")
    print("-" * 65)
    print(f"HELD-OUT TEST PERFORMANCE (Final Unbiased Evaluation):")
    print(f"  Accuracy:            {test_metrics.accuracy * 100:.2f}%")
    print(f"  F1 Score (Macro):    {test_metrics.f1_macro:.4f}")
    if test_metrics.roc_auc is not None:
        print(f"  ROC-AUC:             {test_metrics.roc_auc:.4f}")
    print(f"  Confusion Matrix:\n    {test_metrics.confusion_matrix}")
    print("-" * 65)
    print(f"Saved Search History:  {history_path}")
    print(f"Saved Best Config:     {best_arch_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

"""Experiment management and persistence service for tracking and comparing research runs."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time
from typing import Any

from src.utils.config_loader import get_project_root


class ExperimentService:
    """Service managing persisted experiment records, comparisons, and research exports."""

    @classmethod
    def get_results_dir(cls) -> Path:
        return get_project_root() / "experiments" / "results"

    @classmethod
    def list_saved_experiments(cls) -> list[dict[str, Any]]:
        """Discover and summarize all saved experiment files in the experiments directory."""
        res_dir = cls.get_results_dir()
        if not res_dir.exists():
            return []

        experiments: list[dict[str, Any]] = []

        # 1. Baseline VQC
        base_file = res_dir / "baseline_vqc_results.json"
        if base_file.exists():
            data = json.load(open(base_file, encoding="utf-8"))
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base_file.stat().st_mtime))
            experiments.append({
                "experiment_id": "exp_baseline_vqc_depth12",
                "title": "Baseline VQC Benchmark (2-Layer Linear)",
                "category": "Quantum Baseline",
                "timestamp": mtime,
                "file_path": str(base_file),
                "val_accuracy": data.get("validation_metrics", {}).get("accuracy"),
                "test_accuracy": data.get("test_metrics", {}).get("accuracy"),
                "circuit_depth": data.get("circuit_metadata", {}).get("circuit_depth"),
                "parameters": data.get("circuit_metadata", {}).get("num_trainable_parameters"),
                "runtime_seconds": data.get("training_time_seconds"),
                "status": "Completed",
            })

        # 2. Evolutionary Best Architecture
        best_file = res_dir / "best_architecture.json"
        if best_file.exists():
            data = json.load(open(best_file, encoding="utf-8"))
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(best_file.stat().st_mtime))
            experiments.append({
                "experiment_id": f"exp_opt_{data.get('arch_id', 'search')}",
                "title": f"Search-Optimized Architecture ({data.get('arch_id', '')})",
                "category": "Quantum Search",
                "timestamp": mtime,
                "file_path": str(best_file),
                "val_accuracy": data.get("val_metrics", {}).get("accuracy"),
                "test_accuracy": data.get("test_metrics", {}).get("accuracy"),
                "circuit_depth": data.get("circuit_metadata", {}).get("circuit_depth"),
                "parameters": data.get("circuit_metadata", {}).get("num_trainable_parameters"),
                "runtime_seconds": data.get("training_time_seconds"),
                "status": "Completed",
            })

        # 3. Evolutionary Search Trajectory
        hist_file = res_dir / "search_history.json"
        if hist_file.exists():
            data = json.load(open(hist_file, encoding="utf-8"))
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hist_file.stat().st_mtime))
            experiments.append({
                "experiment_id": "exp_evolutionary_search_run1",
                "title": "Evolutionary Search (Pop=4, Gen=2)",
                "category": "Evolutionary Optimization",
                "timestamp": mtime,
                "file_path": str(hist_file),
                "val_accuracy": data.get("best_candidate", {}).get("val_metrics", {}).get("accuracy"),
                "test_accuracy": None,
                "circuit_depth": data.get("best_candidate", {}).get("circuit_metadata", {}).get("circuit_depth"),
                "parameters": data.get("best_candidate", {}).get("circuit_metadata", {}).get("num_trainable_parameters"),
                "runtime_seconds": data.get("total_runtime_seconds"),
                "status": "Completed",
            })

        # 4. Noise Robustness Benchmarks
        noise_file = res_dir / "noise_experiment_results.json"
        if noise_file.exists():
            data = json.load(open(noise_file, encoding="utf-8"))
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(noise_file.stat().st_mtime))
            opt_metrics = data.get("optimized_architecture", {}).get("test_metrics", {})
            test_acc = opt_metrics.get("accuracy")
            if test_acc is None:
                benchmarks_list = data.get("benchmarks", {}).get("optimized", [])
                if benchmarks_list:
                    test_acc = benchmarks_list[0].get("ideal_metrics", {}).get("accuracy")
            experiments.append({
                "experiment_id": "exp_noise_robustness_nisq",
                "title": "Hardware Noise Robustness (Low, Med, High)",
                "category": "Noise Simulation",
                "timestamp": mtime,
                "file_path": str(noise_file),
                "val_accuracy": None,
                "test_accuracy": test_acc,
                "circuit_depth": None,
                "parameters": None,
                "runtime_seconds": None,
                "status": "Completed",
            })

        return experiments

    @classmethod
    def load_baseline_results(cls) -> dict[str, Any]:
        """Load baseline VQC results from disk."""
        base_file = cls.get_results_dir() / "baseline_vqc_results.json"
        if not base_file.exists():
            return {}
        with open(base_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_raw_experiment_file(cls, path_str: str) -> dict[str, Any]:
        """Read arbitrary saved experiment JSON file safely."""
        path = Path(path_str)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def export_records_to_csv(records: list[dict[str, Any]]) -> str:
        """Convert a list of experiment or benchmark dictionaries to CSV text."""
        if not records:
            return ""
        output = io.StringIO()
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        return output.getvalue()

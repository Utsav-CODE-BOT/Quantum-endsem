"""Visualization routines for evolutionary architecture search dynamics and Pareto trade-offs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def plot_fitness_convergence(search_history: dict[str, Any] | list[dict[str, Any]], save_path: str | Path) -> Path:
    """Plot generational trajectory of best and mean fitness scores."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(search_history, dict):
        history_list = search_history.get("generation_history", [])
    else:
        history_list = search_history

    generations = [h["generation"] for h in history_list]
    best_fitness = [h["best_fitness"] for h in history_list]
    mean_fitness = [h.get("avg_fitness", h.get("mean_fitness", 0.0)) for h in history_list]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(generations, best_fitness, marker="o", color="#1f77b4", linewidth=2.5, label="Best Fitness (Elitist)")
    ax.plot(generations, mean_fitness, marker="s", linestyle="--", color="#ff7f0e", linewidth=2.0, label="Mean Population Fitness")

    ax.set_title("Evolutionary Architecture Search Convergence", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Multi-Objective Fitness", fontsize=12)
    ax.set_xticks(generations)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved search convergence plot to '%s'", save_path)
    return save_path


def plot_pareto_front(
    candidates: Sequence[dict[str, Any]],
    baseline: dict[str, Any] | None,
    optimized: dict[str, Any] | None,
    save_path: str | Path,
) -> Path:
    """Plot Accuracy vs Quantum Circuit Depth (Pareto trade-off)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    # Plot all evaluated candidates
    depths = [c.get("circuit_metadata", {}).get("circuit_depth", 0) for c in candidates]
    accs = [c.get("val_metrics", {}).get("accuracy", 0.0) * 100 for c in candidates]
    ax.scatter(depths, accs, color="#7f7f7f", alpha=0.6, s=60, edgecolors="none", label="Evaluated Architectures")

    # Highlight baseline
    if baseline:
        b_depth = baseline.get("circuit_depth", 12)
        b_acc = baseline.get("val_accuracy", 0.95) * 100
        ax.scatter([b_depth], [b_acc], color="#d62728", s=140, marker="X", zorder=5, label=f"Baseline VQC (Depth {b_depth})")

    # Highlight search optimized
    if optimized:
        o_depth = optimized.get("circuit_metadata", {}).get("circuit_depth", 5)
        o_acc = optimized.get("val_metrics", {}).get("accuracy", 0.95) * 100
        ax.scatter([o_depth], [o_acc], color="#2ca02c", s=180, marker="*", zorder=5, label=f"Search-Optimized VQC (Depth {o_depth})")

    ax.set_title("Pareto Trade-off: Accuracy vs Circuit Depth", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Circuit Depth (lower is better)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved Pareto front plot to '%s'", save_path)
    return save_path

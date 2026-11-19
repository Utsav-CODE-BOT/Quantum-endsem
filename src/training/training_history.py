"""Training history tracker for VQC optimization runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingHistory:
    """Stores loss trajectory, evaluation metrics, and runtime performance."""

    loss_history: list[float] = field(default_factory=list)
    val_accuracy_history: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_weights: list[float] = field(default_factory=list)
    total_evaluations: int = 0
    training_time_seconds: float = 0.0
    converged: bool = False
    stop_reason: str = ""

    def add_step(self, loss_val: float, weights: np.ndarray | None = None, elapsed: float = 0.0) -> None:
        """Record an optimization step."""
        loss_float = float(loss_val)
        self.loss_history.append(loss_float)
        self.timestamps.append(float(elapsed))
        self.total_evaluations += 1

        if loss_float < self.best_loss:
            self.best_loss = loss_float
            if weights is not None:
                self.best_weights = [float(w) for w in weights]

    def to_dict(self) -> dict[str, Any]:
        """Convert history to JSON-serializable dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        """Load history from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

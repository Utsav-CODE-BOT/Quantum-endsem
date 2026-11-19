"""Classical Neural Network (MLP) baseline with parameter count comparable to VQC."""

from __future__ import annotations

import time
from typing import Any
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.evaluation.classification_metrics import (
    ClassificationReport,
    compute_classification_metrics,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClassicalMLP:
    """Multi-Layer Perceptron baseline configured with parameter capacity comparable to VQC."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (4,),
        activation: str = "relu",
        alpha: float = 0.001,
        max_iter: int = 300,
        random_seed: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_seed = random_seed

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_seed,
            solver="adam",
            early_stopping=False,
        )
        self.is_fitted = False
        self.training_time_seconds: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalMLP":
        """Fit MLP classifier on training features."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        t0 = time.perf_counter()
        self.model.fit(X_arr, y_arr)
        self.training_time_seconds = time.perf_counter() - t0
        self.is_fitted = True

        logger.debug(
            "Fitted MLP %s on %d samples in %.4fs (loss: %.4f)",
            self.hidden_layer_sizes,
            len(X_arr),
            self.training_time_seconds,
            self.model.loss_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        if not self.is_fitted:
            raise RuntimeError("MLP must be fitted before predict.")
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("MLP must be fitted before predict_proba.")
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassificationReport:
        """Evaluate model and return standardized ClassificationReport."""
        preds = self.predict(X)
        probas = self.predict_proba(X)
        return compute_classification_metrics(np.asarray(y, dtype=int), preds, probas)

    @property
    def num_parameters(self) -> int:
        """Total trainable weights and biases in the MLP."""
        if not self.is_fitted:
            return 0
        total = sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)
        return int(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "Classical_MLP",
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "activation": self.activation,
            "alpha": self.alpha,
            "num_parameters": self.num_parameters,
            "training_time_seconds": round(self.training_time_seconds, 4),
        }

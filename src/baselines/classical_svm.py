"""Classical Support Vector Machine (Linear and RBF) baselines on reduced PCA features."""

from __future__ import annotations

import time
from typing import Any, Literal
import numpy as np
from sklearn.svm import SVC

from src.evaluation.classification_metrics import (
    ClassificationReport,
    compute_classification_metrics,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClassicalSVM:
    """Scikit-learn Support Vector Machine baseline matching quantum feature inputs."""

    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        probability: bool = True,
        random_seed: int = 42,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_seed = random_seed

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_seed,
        )
        self.is_fitted = False
        self.training_time_seconds: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalSVM":
        """Fit SVM model on training features and binary labels."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        t0 = time.perf_counter()
        self.model.fit(X_arr, y_arr)
        self.training_time_seconds = time.perf_counter() - t0
        self.is_fitted = True

        logger.debug(
            "Fitted SVM (%s, C=%.2f) on %d samples in %.4fs",
            self.kernel,
            self.C,
            len(X_arr),
            self.training_time_seconds,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before predict.")
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("SVM must be fitted before predict_proba.")
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassificationReport:
        """Evaluate model and return standardized ClassificationReport."""
        preds = self.predict(X)
        probas = self.predict_proba(X)
        return compute_classification_metrics(np.asarray(y, dtype=int), preds, probas)

    @property
    def num_parameters(self) -> int:
        """Effective parameter count: support vectors * dimension + intercepts."""
        if not self.is_fitted:
            return 0
        n_sv = self.model.support_vectors_.shape[0]
        n_features = self.model.support_vectors_.shape[1]
        return (n_sv * n_features) + len(self.model.intercept_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": f"SVM_{self.kernel.upper()}",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": str(self.gamma),
            "num_parameters": self.num_parameters,
            "training_time_seconds": round(self.training_time_seconds, 4),
        }

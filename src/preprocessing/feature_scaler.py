"""Feature scaler mapping continuous PCA features into quantum rotation angle ranges."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureScaler:
    """Scales reduced features to a bounded range (default [0, pi]) for angle encoding."""

    def __init__(
        self,
        target_range: tuple[float, float] = (0.0, np.pi),
    ) -> None:
        """Initialize scaler.

        Args:
            target_range: Tuple (min_val, max_val). Default: (0.0, pi).
        """
        self.target_range = target_range
        self.scaler = MinMaxScaler(feature_range=target_range)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "FeatureScaler":
        """Fit scaler strictly on training features.

        Args:
            X_train: Training features array (N, n_features).

        Returns:
            self
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
        logger.info(
            "Fitting FeatureScaler with target range %s on training features...",
            self.target_range,
        )
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to the target range using parameters from training fit.

        Args:
            X: Features array (N, n_features).

        Returns:
            Scaled array with values clipped to target_range.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transforming.")
        scaled = self.scaler.transform(X)
        # Numerical safeguard against small floating point drift
        min_val, max_val = self.target_range
        return np.clip(scaled, min_val, max_val)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return scaled training features."""
        self.fit(X_train)
        return self.transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return scaler metadata for serialization."""
        if not self.is_fitted:
            return {"is_fitted": False, "target_range": list(self.target_range)}
        return {
            "is_fitted": True,
            "target_range": list(self.target_range),
            "data_min": self.scaler.data_min_.tolist(),
            "data_max": self.scaler.data_max_.tolist(),
            "data_range": self.scaler.data_range_.tolist(),
        }

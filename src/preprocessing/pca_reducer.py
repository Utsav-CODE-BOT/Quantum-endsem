"""Principal Component Analysis dimensionality reducer for Fashion-MNIST.

Scientifically rigorous implementation:
- PCA must be fitted STRICTLY on training data only.
- Validation and test sets are transformed using the pre-fitted projection.
- Explained variance ratios and cumulative variance are tracked for reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PCAReducer:
    """Reduces 784-dimensional flattened images to low-dimensional quantum feature vectors."""

    def __init__(
        self,
        n_components: int = 4,
        whiten: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initialize PCA reducer.

        Args:
            n_components: Number of principal components (e.g., 4, 6, 8).
            whiten: Whether to whiten the components.
            random_state: Random state for deterministic solver.
        """
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "PCAReducer":
        """Fit PCA strictly on training data.

        Args:
            X_train: Training feature array of shape (N_train, 784).

        Returns:
            self
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D (N, features), got {X_train.shape}")
        if X_train.shape[1] < self.n_components:
            raise ValueError(
                f"Cannot fit {self.n_components} components with only {X_train.shape[1]} features"
            )

        logger.info(
            "Fitting PCA with n_components=%d on training data (shape=%s)...",
            self.n_components,
            X_train.shape,
        )
        self.pca.fit(X_train)
        self.is_fitted = True

        cum_var = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info(
            "PCA fit complete. Explained variance ratios: %s (Cumulative: %.4f)",
            np.round(self.pca.explained_variance_ratio_, 4).tolist(),
            cum_var,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted PCA projection.

        Args:
            X: Feature array of shape (N, 784).

        Returns:
            Projected array of shape (N, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError("PCAReducer must be fitted on training data before transforming.")
        return self.pca.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit PCA on training data and return the transformed training features."""
        self.fit(X_train)
        return self.transform(X_train)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Explained variance ratio per component."""
        if not self.is_fitted:
            raise RuntimeError("PCAReducer is not fitted.")
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> float:
        """Total cumulative variance explained by the retained components."""
        return float(np.sum(self.explained_variance_ratio))

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for experiment reproducibility."""
        if not self.is_fitted:
            return {"is_fitted": False, "n_components": self.n_components}
        return {
            "is_fitted": True,
            "n_components": self.n_components,
            "whiten": self.whiten,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_variance": self.cumulative_variance,
            "mean": self.pca.mean_.tolist(),
            "components_shape": list(self.pca.components_.shape),
        }

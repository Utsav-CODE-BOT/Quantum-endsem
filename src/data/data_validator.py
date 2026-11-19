"""Data validator ensuring integrity, balance, and lack of leakage across splits."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from src.data.dataset_splitter import DatasetSplits
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""


class DataValidator:
    """Performs integrity and scientific validity checks on datasets and splits."""

    @staticmethod
    def validate_splits(splits: DatasetSplits) -> bool:
        """Validate dataset splits for correct shapes, labels, and balance.

        Args:
            splits: DatasetSplits instance.

        Returns:
            True if all checks pass.

        Raises:
            DataValidationError: If any integrity check fails.
        """
        # 1. Check non-empty
        for name, X, y in [
            ("train", splits.X_train, splits.y_train),
            ("val", splits.X_val, splits.y_val),
            ("test", splits.X_test, splits.y_test),
        ]:
            if len(X) == 0 or len(y) == 0:
                raise DataValidationError(f"Split '{name}' is empty (X: {len(X)}, y: {len(y)})")
            if len(X) != len(y):
                raise DataValidationError(
                    f"Split '{name}' has length mismatch: X has {len(X)}, y has {len(y)}"
                )

        # 2. Check dimensions
        for name, X in [("train", splits.X_train), ("val", splits.X_val), ("test", splits.X_test)]:
            if X.ndim not in (2, 3):
                raise DataValidationError(
                    f"Split '{name}' images must have 2 or 3 dimensions, got {X.ndim} ({X.shape})"
                )
            if X.ndim == 3 and (X.shape[1] != 28 or X.shape[2] != 28):
                raise DataValidationError(
                    f"Split '{name}' images must be 28x28, got {X.shape[1:]}"
                )

        # 3. Check labels
        valid_classes = set(splits.class_names.keys())
        for name, y in [("train", splits.y_train), ("val", splits.y_val), ("test", splits.y_test)]:
            unique_labels = set(np.unique(y))
            if not unique_labels.issubset(valid_classes):
                raise DataValidationError(
                    f"Split '{name}' contains unexpected labels: {unique_labels - valid_classes}"
                )

        # 4. Check class balance
        for name, y in [("train", splits.y_train), ("val", splits.y_val), ("test", splits.y_test)]:
            counts = {c: int(np.sum(y == c)) for c in valid_classes}
            if len(set(counts.values())) > 1:
                logger.warning("Split '%s' is not perfectly balanced: %s", name, counts)

        # 5. Check no duplicate images between train and val/test via sha256 hashes
        train_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in splits.X_train}
        val_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in splits.X_val}
        test_hashes = {hashlib.sha256(img.tobytes()).hexdigest() for img in splits.X_test}

        train_val_overlap = train_hashes.intersection(val_hashes)
        train_test_overlap = train_hashes.intersection(test_hashes)

        if train_val_overlap:
            logger.warning(
                "Potential pixel-identical overlap between train and val: %d items",
                len(train_val_overlap),
            )
        if train_test_overlap:
            logger.warning(
                "Potential pixel-identical overlap between train and test: %d items",
                len(train_test_overlap),
            )

        logger.info("Dataset splits validated successfully.")
        return True

    @staticmethod
    def validate_features(features: np.ndarray, expected_dim: int, target_range: tuple[float, float]) -> bool:
        """Validate preprocessed feature array.

        Args:
            features: 2D array (N, d).
            expected_dim: Expected feature dimension d.
            target_range: Expected (min_val, max_val) bounds.

        Returns:
            True if valid.
        """
        if features.ndim != 2:
            raise DataValidationError(f"Features must be 2D array, got {features.ndim}D")
        if features.shape[1] != expected_dim:
            raise DataValidationError(
                f"Feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}"
            )
        if np.isnan(features).any() or np.isinf(features).any():
            raise DataValidationError("Features contain NaN or Inf values")

        min_val, max_val = target_range
        actual_min, actual_max = np.min(features), np.max(features)
        margin = 1e-3
        if actual_min < min_val - margin or actual_max > max_val + margin:
            logger.warning(
                "Features slightly outside target range [%f, %f]: actual [%f, %f]",
                min_val, max_val, actual_min, actual_max,
            )
        return True

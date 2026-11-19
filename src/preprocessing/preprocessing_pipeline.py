"""Unified end-to-end preprocessing pipeline for Fashion-MNIST to quantum angles.

Orchestrates:
1. Normalization & flattening.
2. Training-only PCA dimensionality reduction.
3. Quantum feature angle scaling.
4. Validation and parameter persistence.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset_splitter import DatasetSplits
from src.data.data_validator import DataValidator
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.pca_reducer import PCAReducer
from src.utils.config_loader import ConfigManager, get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessedSplits:
    """Preprocessed quantum-ready features and labels for all dataset partitions."""

    X_train: np.ndarray  # Shape: (N_train, n_components), range: [0, pi]
    y_train: np.ndarray  # Shape: (N_train,)
    X_val: np.ndarray  # Shape: (N_val, n_components), range: [0, pi]
    y_val: np.ndarray  # Shape: (N_val,)
    X_test: np.ndarray  # Shape: (N_test, n_components), range: [0, pi]
    y_test: np.ndarray  # Shape: (N_test,)
    n_components: int
    target_range: tuple[float, float]
    label_mapping: dict[int, int]
    class_names: dict[int, str]


class PreprocessingPipeline:
    """Complete preprocessing pipeline from raw 28x28 images to quantum angle vectors."""

    def __init__(
        self,
        n_components: int = 4,
        target_range: tuple[float, float] = (0.0, np.pi),
        random_state: int | None = 42,
    ) -> None:
        """Initialize preprocessing pipeline.

        Args:
            n_components: Number of PCA components (quantum features).
            target_range: Bounds for angle scaling.
            random_state: Random state for PCA reproducibility.
        """
        self.n_components = n_components
        self.target_range = target_range
        self.random_state = random_state

        self.image_processor = ImageProcessor()
        self.pca_reducer = PCAReducer(
            n_components=n_components,
            random_state=random_state,
        )
        self.feature_scaler = FeatureScaler(target_range=target_range)
        self.is_fitted = False

    def fit(self, X_train_raw: np.ndarray) -> "PreprocessingPipeline":
        """Fit all preprocessing stages strictly on raw training images.

        Args:
            X_train_raw: Raw training images (N_train, 28, 28) or (N_train, 784).

        Returns:
            self
        """
        # 1. Process images
        X_train_flat = self.image_processor.process(X_train_raw)

        # 2. Fit PCA on training images only
        X_train_pca = self.pca_reducer.fit_transform(X_train_flat)

        # 3. Fit angle scaler on training PCA features only
        self.feature_scaler.fit(X_train_pca)

        self.is_fitted = True
        logger.info("PreprocessingPipeline successfully fitted on %d samples.", len(X_train_raw))
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Transform raw images into quantum rotation angle vectors.

        Args:
            X_raw: Array of shape (N, 28, 28) or (N, 784).

        Returns:
            Angles array of shape (N, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")

        X_flat = self.image_processor.process(X_raw)
        X_pca = self.pca_reducer.transform(X_flat)
        X_angles = self.feature_scaler.transform(X_pca)

        DataValidator.validate_features(
            X_angles,
            expected_dim=self.n_components,
            target_range=self.target_range,
        )
        return X_angles

    def transform_splits(self, splits: DatasetSplits) -> PreprocessedSplits:
        """Fit pipeline on train split and transform all splits.

        Enforces zero data leakage:
        - PCA & Scaler fit exclusively on splits.X_train.
        - splits.X_val and splits.X_test are mapped via pre-fitted transformers.

        Args:
            splits: Raw DatasetSplits.

        Returns:
            PreprocessedSplits ready for quantum circuit feature encoding.
        """
        DataValidator.validate_splits(splits)

        logger.info("Fitting preprocessing pipeline on training split...")
        self.fit(splits.X_train)

        X_train_enc = self.transform(splits.X_train)
        X_val_enc = self.transform(splits.X_val)
        X_test_enc = self.transform(splits.X_test)

        logger.info(
            "Transformed splits: train=%s, val=%s, test=%s (feature_dim=%d)",
            X_train_enc.shape,
            X_val_enc.shape,
            X_test_enc.shape,
            self.n_components,
        )

        return PreprocessedSplits(
            X_train=X_train_enc,
            y_train=splits.y_train,
            X_val=X_val_enc,
            y_val=splits.y_val,
            X_test=X_test_enc,
            y_test=splits.y_test,
            n_components=self.n_components,
            target_range=self.target_range,
            label_mapping=splits.label_mapping,
            class_names=splits.class_names,
        )

    def save(self, path: str | Path) -> None:
        """Serialize fitted pipeline to disk.

        Args:
            path: Destination file path (.pkl).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted PreprocessingPipeline.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved PreprocessingPipeline to '%s'", path)

    @staticmethod
    def load(path: str | Path) -> "PreprocessingPipeline":
        """Load fitted pipeline from disk.

        Args:
            path: Source file path.

        Returns:
            Fitted PreprocessingPipeline instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info("Loaded PreprocessingPipeline from '%s'", path)
        return pipeline

    def get_metadata(self) -> dict[str, Any]:
        """Return pipeline parameters and PCA variance explained."""
        return {
            "n_components": self.n_components,
            "target_range": list(self.target_range),
            "random_state": self.random_state,
            "pca_metadata": self.pca_reducer.get_metadata(),
            "scaler_metadata": self.feature_scaler.get_metadata(),
        }

    @classmethod
    def from_config(cls, config: ConfigManager) -> "PreprocessingPipeline":
        """Factory method to instantiate from configuration."""
        n_comp = config.get("preprocessing", "preprocessing.pca.n_components", 4)
        target = config.get(
            "preprocessing",
            "preprocessing.feature_scaling.target_range",
            [0.0, float(np.pi)],
        )
        seed = config.get("training", "training.parameter_init_seed", 42)
        return cls(
            n_components=n_comp,
            target_range=(float(target[0]), float(target[1])),
            random_state=seed,
        )

"""Dataset service adapting Fashion-MNIST data ingestion and preprocessing to the UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from src.data.dataset_loader import FashionMNISTLoader, RawDataset
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.utils.config_loader import get_project_root


class DatasetService:
    """Service providing dataset inspection and preprocessing inspection for the UI."""

    _cached_raw_train: RawDataset | None = None
    _cached_raw_test: RawDataset | None = None
    _cached_pipeline: PreprocessingPipeline | None = None

    @classmethod
    def get_raw_dataset(cls, split: str = "train") -> RawDataset:
        """Load and cache raw Fashion-MNIST split."""
        if split == "train":
            if cls._cached_raw_train is None:
                loader = FashionMNISTLoader(selected_classes=(0, 9))
                cls._cached_raw_train = loader.load(split="train")
            return cls._cached_raw_train
        else:
            if cls._cached_raw_test is None:
                loader = FashionMNISTLoader(selected_classes=(0, 9))
                cls._cached_raw_test = loader.load(split="test")
            return cls._cached_raw_test

    @classmethod
    def get_quantum_ready_data(cls) -> dict[str, np.ndarray]:
        """Load preprocessed quantum-ready NPZ arrays."""
        path = get_project_root() / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
        if not path.exists():
            raise FileNotFoundError(f"Quantum-ready dataset not found at {path}")
        return dict(np.load(path))

    @classmethod
    def get_pca_metadata(cls) -> dict[str, Any]:
        """Load PCA dimensionality reduction metadata and explained variance."""
        path = get_project_root() / "data" / "metadata" / "preprocessing_metadata.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_pipeline(cls) -> PreprocessingPipeline:
        """Load the pre-fitted preprocessing pipeline."""
        if cls._cached_pipeline is None:
            path = get_project_root() / "data" / "metadata" / "preprocessing_pipeline.pkl"
            cls._cached_pipeline = PreprocessingPipeline.load(path)
        return cls._cached_pipeline

    @classmethod
    def inspect_sample_pipeline(cls, sample_idx: int = 0, split: str = "train") -> dict[str, Any]:
        """Extract step-by-step transformations for a specific image without data leakage.

        Stages:
        1. Original 28x28 image
        2. Normalized flattened 784 vector
        3. PCA reduced 4-component vector
        4. Scaled [0, pi] quantum angle vector
        """
        raw = cls.get_raw_dataset(split)
        if sample_idx >= len(raw.images):
            sample_idx = 0

        img = raw.images[sample_idx]
        orig_label = int(raw.original_labels[sample_idx])
        class_name = "T-shirt/top" if orig_label == 0 else "Ankle boot"

        pipeline = cls.get_pipeline()

        # Step 1: Flatten & normalize
        norm_flat = pipeline.image_processor.process(img[np.newaxis, ...])[0]

        # Step 2: Transform via fitted PCA
        pca_feat = pipeline.pca_reducer.transform(norm_flat[np.newaxis, ...])[0]

        # Step 3: Transform via fitted FeatureScaler
        scaled_angles = pipeline.feature_scaler.transform(pca_feat[np.newaxis, ...])[0]

        return {
            "index": sample_idx,
            "original_label": orig_label,
            "class_name": class_name,
            "raw_image": img,
            "normalized_flat": norm_flat,
            "pca_features": pca_feat,
            "scaled_angles": scaled_angles,
        }

    @classmethod
    def get_dataset_summary(cls) -> dict[str, Any]:
        """Return dataset partition sizes, dimensions, and classes."""
        data = cls.get_quantum_ready_data()
        meta = cls.get_pca_metadata()

        return {
            "name": "Fashion-MNIST (Binary Classification Subproblem)",
            "selected_classes": {0: "T-shirt/top", 9: "Ankle boot"},
            "raw_dimensions": "28 × 28 pixels (Grayscale)",
            "train_samples": len(data["X_train"]),
            "val_samples": len(data["X_val"]),
            "test_samples": len(data["X_test"]),
            "total_preprocessed": len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]),
            "num_pca_components": meta.get("n_components", 4),
            "explained_variance_ratio": meta.get("pca_metadata", {}).get("explained_variance_ratio", []),
            "cumulative_variance": meta.get("pca_metadata", {}).get("cumulative_variance", 0.0),
            "angle_range": "[0, pi] radians",
        }

"""Unit and leakage-prevention tests for the preprocessing pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset_splitter import DatasetSplits
from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.pca_reducer import PCAReducer
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


class TestImageProcessor:
    """Test normalization and flattening."""

    def test_normalization_range(self) -> None:
        processor = ImageProcessor()
        raw = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
        norm = processor.normalize(raw)
        assert norm.dtype == np.float32
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_flatten_shape(self) -> None:
        processor = ImageProcessor()
        raw = np.zeros((15, 28, 28), dtype=np.float32)
        flat = processor.flatten(raw)
        assert flat.shape == (15, 784)


class TestPCALeakagePrevention:
    """Rigorous scientific test verifying zero data leakage in PCA."""

    def test_pca_fit_strictly_on_train(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 784))
        X_val_1 = rng.normal(loc=10.0, scale=2.0, size=(20, 784))
        X_val_2 = rng.normal(loc=-50.0, scale=5.0, size=(20, 784))

        pca = PCAReducer(n_components=4, random_state=42)
        pca.fit(X_train)

        # Baseline training components
        components_train = pca.pca.components_.copy()
        mean_train = pca.pca.mean_.copy()

        # Transforming val_1 vs val_2 must NOT mutate the fitted PCA attributes
        _ = pca.transform(X_val_1)
        np.testing.assert_array_equal(pca.pca.components_, components_train)
        np.testing.assert_array_equal(pca.pca.mean_, mean_train)

        _ = pca.transform(X_val_2)
        np.testing.assert_array_equal(pca.pca.components_, components_train)
        np.testing.assert_array_equal(pca.pca.mean_, mean_train)

    def test_unfitted_transform_raises(self) -> None:
        pca = PCAReducer(n_components=4)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pca.transform(np.zeros((5, 784)))


class TestFeatureScaler:
    """Test feature scaling to quantum angle bounds."""

    def test_scaling_to_pi(self) -> None:
        scaler = FeatureScaler(target_range=(0.0, np.pi))
        X = np.array([[-10.0, 5.0], [0.0, 15.0], [10.0, 25.0]])
        scaled = scaler.fit_transform(X)

        assert scaled.shape == X.shape
        assert scaled.min() >= 0.0 - 1e-6
        assert scaled.max() <= np.pi + 1e-6


class TestPreprocessingPipeline:
    """Test end-to-end preprocessing pipeline and serialization."""

    def test_pipeline_transform_splits(self) -> None:
        rng = np.random.default_rng(42)
        splits = DatasetSplits(
            X_train=rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8),
            y_train=np.array([0, 1] * 20),
            X_val=rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8),
            y_val=np.array([0, 1] * 5),
            X_test=rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8),
            y_test=np.array([0, 1] * 5),
            label_mapping={0: 0, 9: 1},
            class_names={0: "T-shirt", 1: "Ankle boot"},
        )

        pipeline = PreprocessingPipeline(n_components=4, random_state=42)
        prep_splits = pipeline.transform_splits(splits)

        assert prep_splits.X_train.shape == (40, 4)
        assert prep_splits.X_val.shape == (10, 4)
        assert prep_splits.X_test.shape == (10, 4)
        assert prep_splits.n_components == 4
        assert 0.0 <= prep_splits.X_train.min()
        assert prep_splits.X_train.max() <= np.pi

    def test_pipeline_serialization(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.integers(0, 256, size=(30, 28, 28), dtype=np.uint8)
        X_test = rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8)

        pipeline = PreprocessingPipeline(n_components=4, random_state=42)
        pipeline.fit(X_train)
        pred_original = pipeline.transform(X_test)

        save_path = tmp_path / "pipeline.pkl"
        pipeline.save(save_path)

        loaded_pipeline = PreprocessingPipeline.load(save_path)
        pred_loaded = loaded_pipeline.transform(X_test)

        np.testing.assert_allclose(pred_original, pred_loaded)

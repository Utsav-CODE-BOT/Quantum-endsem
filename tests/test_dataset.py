"""Unit and integration tests for Fashion-MNIST loading and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset_loader import FashionMNISTLoader, RawDataset
from src.data.dataset_splitter import DatasetSplitter, DatasetSplits
from src.data.data_validator import DataValidationError, DataValidator
from src.utils.seed_manager import SeedManager


class TestDatasetLoader:
    """Test loading and filtering Fashion-MNIST."""

    def test_loader_binary_classes(self) -> None:
        loader = FashionMNISTLoader(selected_classes=[0, 9])
        train_data = loader.load(split="train")

        assert isinstance(train_data, RawDataset)
        assert train_data.images.ndim == 3
        assert train_data.images.shape[1:] == (28, 28)
        assert len(train_data.images) == len(train_data.labels)
        # Check remapped labels are strictly {0, 1}
        assert set(np.unique(train_data.labels)) == {0, 1}
        # Check original labels match requested classes
        assert set(np.unique(train_data.original_labels)) == {0, 9}

    def test_loader_label_mapping_integrity(self) -> None:
        loader = FashionMNISTLoader(selected_classes=[1, 8])
        train_data = loader.load(split="train")
        assert loader.label_mapping == {1: 0, 8: 1}
        assert train_data.class_names[0] == "Trouser"
        assert train_data.class_names[1] == "Bag"


class TestDatasetSplitter:
    """Test balanced splitting and reproducibility."""

    def test_splitter_sample_counts_and_balance(self) -> None:
        loader = FashionMNISTLoader(selected_classes=[0, 9])
        train_data = loader.load(split="train")
        test_data = loader.load(split="test")

        n_train = 50
        n_val = 20
        n_test = 30
        splitter = DatasetSplitter(
            samples_per_class_train=n_train,
            samples_per_class_val=n_val,
            samples_per_class_test=n_test,
            seed_manager=SeedManager(global_seed=42),
        )

        splits = splitter.create_splits(train_data, test_data)
        DataValidator.validate_splits(splits)

        # 2 classes -> exact totals
        assert len(splits.X_train) == n_train * 2
        assert len(splits.X_val) == n_val * 2
        assert len(splits.X_test) == n_test * 2

        # Check each class has exact count
        for c in [0, 1]:
            assert np.sum(splits.y_train == c) == n_train
            assert np.sum(splits.y_val == c) == n_val
            assert np.sum(splits.y_test == c) == n_test

    def test_splits_reproducibility(self) -> None:
        loader = FashionMNISTLoader(selected_classes=[0, 9])
        train_data = loader.load(split="train")

        splitter_a = DatasetSplitter(
            samples_per_class_train=20,
            samples_per_class_val=10,
            samples_per_class_test=10,
            seed_manager=SeedManager(global_seed=999),
        )
        splits_a = splitter_a.create_splits(train_data)

        splitter_b = DatasetSplitter(
            samples_per_class_train=20,
            samples_per_class_val=10,
            samples_per_class_test=10,
            seed_manager=SeedManager(global_seed=999),
        )
        splits_b = splitter_b.create_splits(train_data)

        np.testing.assert_array_equal(splits_a.X_train, splits_b.X_train)
        np.testing.assert_array_equal(splits_a.y_train, splits_b.y_train)


class TestDataValidator:
    """Test validation and sanity checks."""

    def test_validator_detects_label_mismatch(self) -> None:
        splits = DatasetSplits(
            X_train=np.zeros((10, 28, 28)),
            y_train=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 5]),  # Label 5 is invalid
            X_val=np.zeros((4, 28, 28)),
            y_val=np.array([0, 1, 0, 1]),
            X_test=np.zeros((4, 28, 28)),
            y_test=np.array([0, 1, 0, 1]),
            label_mapping={0: 0, 9: 1},
            class_names={0: "T-shirt", 1: "Boot"},
        )
        with pytest.raises(DataValidationError, match="unexpected labels"):
            DataValidator.validate_splits(splits)

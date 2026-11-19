"""Balanced train, validation, and test dataset partitioner.

Constructs strictly disjoint, balanced subsets per class for
experimentation and benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.dataset_loader import RawDataset
from src.utils.logging_config import get_logger
from src.utils.seed_manager import SeedManager

logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    """Container for partitioned dataset splits."""

    X_train: np.ndarray  # Shape: (N_train, 28, 28)
    y_train: np.ndarray  # Shape: (N_train,)
    X_val: np.ndarray  # Shape: (N_val, 28, 28)
    y_val: np.ndarray  # Shape: (N_val,)
    X_test: np.ndarray  # Shape: (N_test, 28, 28)
    y_test: np.ndarray  # Shape: (N_test,)
    label_mapping: dict[int, int]
    class_names: dict[int, str]

    def summary(self) -> dict[str, int]:
        """Return sample counts per split."""
        return {
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "total_samples": len(self.X_train) + len(self.X_val) + len(self.X_test),
        }


class DatasetSplitter:
    """Partitions dataset into balanced training, validation, and test subsets."""

    def __init__(
        self,
        samples_per_class_train: int = 100,
        samples_per_class_val: int = 30,
        samples_per_class_test: int = 50,
        seed_manager: SeedManager | None = None,
    ) -> None:
        """Initialize splitter.

        Args:
            samples_per_class_train: Number of training samples per class.
            samples_per_class_val: Number of validation samples per class.
            samples_per_class_test: Number of test samples per class.
            seed_manager: SeedManager for deterministic sampling.
        """
        self.n_train = samples_per_class_train
        self.n_val = samples_per_class_val
        self.n_test = samples_per_class_test
        self.seed_manager = seed_manager or SeedManager(global_seed=42)
        self.rng = self.seed_manager.get_numpy_rng("dataset_splitter")

    def create_splits(
        self,
        train_raw: RawDataset,
        test_raw: RawDataset | None = None,
    ) -> DatasetSplits:
        """Create balanced train, validation, and test splits.

        Train and validation are drawn from the official training split.
        Test is drawn from the official test split (if provided) or from
        held-out training split data.

        Args:
            train_raw: RawDataset loaded from official training data.
            test_raw: Optional RawDataset loaded from official test data.

        Returns:
            DatasetSplits containing balanced and disjoint subsets.
        """
        classes = sorted(list(train_raw.class_names.keys()))

        train_indices_list: list[int] = []
        val_indices_list: list[int] = []
        test_indices_list: list[int] = []

        # 1. Sample train and val from train_raw
        for c in classes:
            class_indices = np.where(train_raw.labels == c)[0]
            total_needed_from_train = self.n_train + self.n_val
            if test_raw is None:
                total_needed_from_train += self.n_test

            if len(class_indices) < total_needed_from_train:
                raise ValueError(
                    f"Class {c} has only {len(class_indices)} samples in train_raw, "
                    f"but {total_needed_from_train} are requested."
                )

            # Permute indices deterministically
            shuffled = self.rng.permutation(class_indices)
            train_idx = shuffled[: self.n_train]
            val_idx = shuffled[self.n_train : self.n_train + self.n_val]
            train_indices_list.extend(train_idx)
            val_indices_list.extend(val_idx)

            if test_raw is None:
                test_idx = shuffled[
                    self.n_train + self.n_val : self.n_train + self.n_val + self.n_test
                ]
                test_indices_list.extend(test_idx)

        # 2. Sample test from test_raw if available
        if test_raw is not None:
            for c in classes:
                class_indices = np.where(test_raw.labels == c)[0]
                if len(class_indices) < self.n_test:
                    raise ValueError(
                        f"Class {c} has only {len(class_indices)} samples in test_raw, "
                        f"but {self.n_test} are requested."
                    )
                shuffled = self.rng.permutation(class_indices)
                test_idx = shuffled[: self.n_test]
                test_indices_list.extend(test_idx)

        # Construct array partitions
        X_train = train_raw.images[train_indices_list]
        y_train = train_raw.labels[train_indices_list]
        X_val = train_raw.images[val_indices_list]
        y_val = train_raw.labels[val_indices_list]

        if test_raw is not None:
            X_test = test_raw.images[test_indices_list]
            y_test = test_raw.labels[test_indices_list]
        else:
            X_test = train_raw.images[test_indices_list]
            y_test = train_raw.labels[test_indices_list]

        # Final shuffle of each split so classes are interspersed
        train_perm = self.rng.permutation(len(X_train))
        val_perm = self.rng.permutation(len(X_val))
        test_perm = self.rng.permutation(len(X_test))

        splits = DatasetSplits(
            X_train=X_train[train_perm],
            y_train=y_train[train_perm],
            X_val=X_val[val_perm],
            y_val=y_val[val_perm],
            X_test=X_test[test_perm],
            y_test=y_test[test_perm],
            label_mapping=train_raw.label_mapping,
            class_names=train_raw.class_names,
        )

        logger.info(
            "Created dataset splits: train=%d, val=%d, test=%d",
            len(splits.X_train),
            len(splits.X_val),
            len(splits.X_test),
        )
        return splits

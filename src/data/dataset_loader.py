"""Fashion-MNIST dataset loader with class filtering and label remapping.

Supports loading official Fashion-MNIST dataset via torchvision,
filtering to arbitrary class subsets, and remapping labels to {0, 1, ...}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torchvision
import torchvision.transforms as transforms

from src.utils.config_loader import ConfigManager, get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

FASHION_MNIST_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


@dataclass
class RawDataset:
    """Container for loaded raw dataset arrays."""

    images: np.ndarray  # Shape: (N, 28, 28), dtype: uint8 or float32
    labels: np.ndarray  # Shape: (N,), dtype: int64 (remapped: 0, 1, ...)
    original_labels: np.ndarray  # Shape: (N,), original Fashion-MNIST labels
    label_mapping: dict[int, int]  # original_label -> remapped_label
    class_names: dict[int, str]  # remapped_label -> class_name


class FashionMNISTLoader:
    """Loads and filters Fashion-MNIST images and labels."""

    def __init__(
        self,
        raw_data_dir: str | Path | None = None,
        selected_classes: Sequence[int] = (0, 9),
    ) -> None:
        """Initialize the loader.

        Args:
            raw_data_dir: Directory where raw dataset is cached.
            selected_classes: List of original class integers to extract.
        """
        if raw_data_dir is None:
            raw_data_dir = get_project_root() / "data" / "raw"
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.is_absolute():
            self.raw_data_dir = get_project_root() / self.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.selected_classes = list(selected_classes)
        # Remap selected classes to 0, 1, ..., C-1
        self.label_mapping = {
            orig: new for new, orig in enumerate(sorted(self.selected_classes))
        }
        self.class_names = {
            self.label_mapping[orig]: FASHION_MNIST_CLASSES[orig]
            for orig in self.selected_classes
        }

        logger.info(
            "FashionMNISTLoader initialized: selected_classes=%s, mapping=%s",
            self.selected_classes,
            self.label_mapping,
        )

    def load(self, split: str = "train") -> RawDataset:
        """Load and filter the requested split of Fashion-MNIST.

        Args:
            split: 'train' (60k original) or 'test' (10k original).

        Returns:
            RawDataset dataclass containing filtered images and remapped labels.
        """
        is_train = split.lower() == "train"
        logger.info("Loading Fashion-MNIST split='%s' from '%s'...", split, self.raw_data_dir)

        # Download and load via torchvision
        dataset = torchvision.datasets.FashionMNIST(
            root=str(self.raw_data_dir),
            train=is_train,
            download=True,
            transform=None,
        )

        # Convert to numpy arrays
        raw_images = dataset.data.numpy()  # shape (N, 28, 28), uint8
        raw_labels = dataset.targets.numpy()  # shape (N,), int64

        # Filter for selected classes
        mask = np.isin(raw_labels, self.selected_classes)
        filtered_images = raw_images[mask]
        filtered_original_labels = raw_labels[mask]

        # Remap labels
        remapped_labels = np.array(
            [self.label_mapping[label] for label in filtered_original_labels],
            dtype=np.int64,
        )

        logger.info(
            "Loaded split='%s': %d samples matching classes %s (out of %d total)",
            split,
            len(filtered_images),
            self.selected_classes,
            len(raw_images),
        )

        return RawDataset(
            images=filtered_images,
            labels=remapped_labels,
            original_labels=filtered_original_labels,
            label_mapping=self.label_mapping,
            class_names=self.class_names,
        )

    @classmethod
    def from_config(cls, config: ConfigManager) -> "FashionMNISTLoader":
        """Factory method to construct loader from configuration."""
        raw_dir = config.get("dataset", "dataset.raw_data_dir", "data/raw")
        classes = config.get("dataset", "dataset.selected_classes", [0, 9])
        return cls(raw_data_dir=raw_dir, selected_classes=classes)

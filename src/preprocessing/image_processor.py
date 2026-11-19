"""Image normalizer and flattener for Fashion-MNIST 28x28 grayscale images."""

from __future__ import annotations

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Normalizes and flattens 28x28 grayscale images."""

    def __init__(self, normalization: str = "minmax") -> None:
        """Initialize image processor.

        Args:
            normalization: 'minmax' scales pixel values from [0, 255] to [0.0, 1.0].
        """
        self.normalization = normalization.lower()
        if self.normalization != "minmax":
            raise ValueError(f"Unsupported normalization method: '{normalization}'")

    def normalize(self, images: np.ndarray) -> np.ndarray:
        """Normalize raw images to [0.0, 1.0] float32.

        Args:
            images: Array of shape (N, 28, 28) or (N, 784), uint8 or float.

        Returns:
            Normalized float32 array in [0.0, 1.0].
        """
        arr = np.asarray(images, dtype=np.float32)
        # Only scale if data appears to be in 0-255 range
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def flatten(self, images: np.ndarray) -> np.ndarray:
        """Flatten 28x28 images into 784-dimensional vectors.

        Args:
            images: Array of shape (N, 28, 28) or (N, 784).

        Returns:
            Array of shape (N, 784).
        """
        arr = np.asarray(images)
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 2:
            return arr
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    def process(self, images: np.ndarray) -> np.ndarray:
        """Apply normalization followed by flattening.

        Args:
            images: Raw images array (N, 28, 28).

        Returns:
            Normalized and flattened array (N, 784) in [0.0, 1.0].
        """
        normalized = self.normalize(images)
        return self.flatten(normalized)

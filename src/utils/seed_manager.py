"""Random seed management for reproducibility.

Centralizes seed control across Python, NumPy, PyTorch, and Qiskit
to ensure reproducible experiments.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SeedManager:
    """Manages random seeds across all libraries for reproducibility.

    Usage:
        seed_mgr = SeedManager(global_seed=42)
        seed_mgr.set_all_seeds()

        # Derive a deterministic seed for a specific component
        component_seed = seed_mgr.derive_seed("architecture_search")
    """

    def __init__(self, global_seed: int = 42) -> None:
        """Initialize the seed manager.

        Args:
            global_seed: The master seed from which all other seeds are derived.
        """
        self._global_seed = global_seed
        self._derived_seeds: dict[str, int] = {}
        logger.info("SeedManager initialized with global_seed=%d", global_seed)

    @property
    def global_seed(self) -> int:
        """The master random seed."""
        return self._global_seed

    def set_all_seeds(self, seed: int | None = None) -> None:
        """Set seeds for Python, NumPy, and optionally PyTorch.

        Args:
            seed: Seed to use. Defaults to the global seed.
        """
        seed = seed if seed is not None else self._global_seed

        # Python stdlib
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch (optional — only if installed and imported)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Ensure deterministic operations where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass  # PyTorch not available

        logger.info("All random seeds set to %d", seed)

    def derive_seed(self, component_name: str) -> int:
        """Derive a deterministic seed for a specific component.

        Uses a hash of the global seed and component name to produce
        a unique, reproducible seed for each component.

        Args:
            component_name: Identifier for the component (e.g., 'pca', 'training').

        Returns:
            A deterministic integer seed for the component.
        """
        if component_name in self._derived_seeds:
            return self._derived_seeds[component_name]

        # Hash the global seed + component name for a deterministic derived seed
        hash_input = f"{self._global_seed}:{component_name}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        derived = int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31)

        self._derived_seeds[component_name] = derived
        logger.debug(
            "Derived seed for '%s': %d (from global_seed=%d)",
            component_name, derived, self._global_seed,
        )
        return derived

    def get_numpy_rng(self, component_name: str) -> np.random.Generator:
        """Create a NumPy random Generator for a specific component.

        Preferred over global seed setting for concurrent or
        component-isolated randomness.

        Args:
            component_name: Identifier for the component.

        Returns:
            A seeded numpy random Generator.
        """
        seed = self.derive_seed(component_name)
        return np.random.default_rng(seed)

    def get_state(self) -> dict:
        """Return the current state for serialization."""
        return {
            "global_seed": self._global_seed,
            "derived_seeds": dict(self._derived_seeds),
        }

    def __repr__(self) -> str:
        return f"SeedManager(global_seed={self._global_seed})"

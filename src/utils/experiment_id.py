"""Unique experiment execution ID generation and trained weight hashing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Sequence
import uuid

import numpy as np


def generate_experiment_id(prefix: str = "EXP") -> str:
    """Generate a unique, timestamped experiment run identifier.

    Format: 'EXP-YYYYMMDD-HHMMSS-<6-hex-id>'
    Example: 'EXP-20261120-164500-a1b2c3'
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{prefix}-{ts}-{uid}"


def compute_weights_hash(weights: np.ndarray | Sequence[float]) -> str:
    """Compute deterministic SHA-256 hash of a numerical weight vector.

    Args:
        weights: Iterable or array of float weights.

    Returns:
        16-character hexadecimal hash string.
    """
    arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

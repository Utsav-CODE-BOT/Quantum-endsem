"""Classification metrics computation for VQC and baseline models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationReport:
    """Standardized evaluation results container."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    roc_auc: float | None = None
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float] | None = None,
) -> ClassificationReport:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted class labels (N,).
        y_prob: Optional predicted class probabilities (N,) or (N, 2).

    Returns:
        ClassificationReport dataclass.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    roc_auc: float | None = None
    if y_prob is not None:
        try:
            prob_arr = np.asarray(y_prob, dtype=float)
            if prob_arr.ndim == 2 and prob_arr.shape[1] == 2:
                prob_arr = prob_arr[:, 1]
            if len(np.unique(y_true_arr)) == 2:
                roc_auc = float(roc_auc_score(y_true_arr, prob_arr))
        except Exception as e:
            logger.debug("ROC-AUC computation skipped or failed: %s", e)

    return ClassificationReport(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
        confusion_matrix=cm,
        roc_auc=roc_auc,
        num_samples=len(y_true_arr),
    )

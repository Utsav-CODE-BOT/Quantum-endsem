"""Measurement and output postprocessing strategies for Variational Quantum Classifiers.

Defines the mathematical mapping from quantum computational basis bitstrings
to classical binary classification labels:
- Parity interpretation: f(x) = sum(b_i) mod 2 in {0, 1}
- First-qubit interpretation: f(x) = b_0 in {0, 1}
"""

from __future__ import annotations

from typing import Callable

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parity_interpret(bitstring_int: int) -> int:
    """Map an integer bitstring representation to its parity (0 or 1).

    Parity = (number of 1-bits) mod 2.
    For example:
        '00' (0) -> 0
        '01' (1) -> 1
        '10' (2) -> 1
        '11' (3) -> 0

    Args:
        bitstring_int: Integer representation of the measurement bitstring.

    Returns:
        0 or 1 class label.
    """
    return bin(bitstring_int).count("1") % 2


def single_qubit_interpret(bitstring_int: int) -> int:
    """Map a bitstring to the state of the lowest-order (first) qubit.

    Args:
        bitstring_int: Integer representation of the measurement bitstring.

    Returns:
        0 or 1.
    """
    return bitstring_int & 1


class MeasurementStrategy:
    """Factory and registry for VQC measurement post-processing functions."""

    SUPPORTED_STRATEGIES = {
        "parity": parity_interpret,
        "single_qubit": single_qubit_interpret,
    }

    @classmethod
    def get_interpret_func(cls, strategy_name: str = "parity") -> Callable[[int], int]:
        """Get the interpretation callable for the given strategy name.

        Args:
            strategy_name: 'parity' or 'single_qubit'.

        Returns:
            Callable mapping integer bitstring to class integer.
        """
        name = strategy_name.lower()
        if name not in cls.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported measurement strategy '{strategy_name}'. "
                f"Supported: {list(cls.SUPPORTED_STRATEGIES.keys())}"
            )
        return cls.SUPPORTED_STRATEGIES[name]

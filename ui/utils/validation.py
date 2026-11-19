"""Input validation utilities for architecture configuration and search hyperparameters."""

from __future__ import annotations


def validate_architecture_params(
    num_qubits: int,
    num_layers: int,
    rotation_gates: list[str],
    entanglement_type: str,
    entanglement_freq: int,
) -> tuple[bool, str]:
    """Validate quantum architecture hyperparameters.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if num_qubits not in [2, 4, 6, 8]:
        return False, f"num_qubits must be 2, 4, 6, or 8, got {num_qubits}"
    if not (1 <= num_layers <= 6):
        return False, f"num_layers must be between 1 and 6, got {num_layers}"
    if not rotation_gates or any(r.lower() not in ("rx", "ry", "rz") for r in rotation_gates):
        return False, f"rotation_gates must contain valid rotations (rx, ry, rz), got {rotation_gates}"
    if entanglement_type not in ("linear", "circular", "pairwise", "none"):
        return False, f"Invalid entanglement_type '{entanglement_type}'"
    if not (1 <= entanglement_freq <= 3):
        return False, f"entanglement_freq must be between 1 and 3, got {entanglement_freq}"
    return True, ""


def validate_fitness_weights(
    w_perf: float,
    w_comp: float,
    w_rob: float,
) -> tuple[bool, str]:
    """Validate that fitness objective weights sum approximately to 1.0."""
    total = w_perf + w_comp + w_rob
    if abs(total - 1.0) > 0.05:
        return False, f"Weights must sum to 1.0 (current sum: {total:.2f})"
    if min(w_perf, w_comp, w_rob) < 0.0:
        return False, "Weights must be non-negative"
    return True, ""


def validate_search_params(
    pop_size: int,
    generations: int,
    tournament_size: int,
) -> tuple[bool, str]:
    """Validate evolutionary search parameters."""
    if pop_size < 2:
        return False, "Population size must be at least 2"
    if generations < 1:
        return False, "Number of generations must be at least 1"
    if tournament_size < 2 or tournament_size > pop_size:
        return False, f"Tournament size must be between 2 and {pop_size}"
    return True, ""

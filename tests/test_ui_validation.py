"""Unit tests for UI validation and formatting utilities."""

from __future__ import annotations

import pytest

from ui.utils.formatting import (
    format_latency,
    format_parameters,
    format_percentage,
    format_time,
)
from ui.utils.validation import (
    validate_architecture_params,
    validate_fitness_weights,
    validate_search_params,
)


class TestFormattingUtils:
    def test_format_percentage(self):
        assert format_percentage(0.9523) == "95.23%"
        assert format_percentage(0.9523, decimals=1) == "95.2%"
        assert format_percentage(1.0) == "100.00%"
        assert format_percentage(0.0) == "0.00%"
        assert format_percentage(None) == "N/A"

    def test_format_time(self):
        assert format_time(45.2) == "45.20 s"
        assert format_time(125.0) == "2m 5.0s"
        assert format_time(0.05) == "50.0 ms"
        assert format_time(None) == "N/A"

    def test_format_latency(self):
        assert format_latency(5.2) == "5.20 ms/sample"
        assert format_latency(0.05) == "50.0 µs/sample"
        assert format_latency(None) == "N/A"

    def test_format_parameters(self):
        assert format_parameters(16) == "16"
        assert format_parameters(1250) == "1,250"
        assert format_parameters(None) == "N/A"


class TestValidationUtils:
    def test_validate_architecture_params_valid(self):
        is_valid, msg = validate_architecture_params(
            num_qubits=4,
            num_layers=2,
            rotation_gates=["ry", "rz"],
            entanglement_type="linear",
            entanglement_freq=1,
        )
        assert is_valid
        assert msg == ""

    def test_validate_architecture_params_invalid_qubits(self):
        is_valid, msg = validate_architecture_params(
            num_qubits=10,  # exceeds allowed [2, 4, 6, 8]
            num_layers=2,
            rotation_gates=["ry"],
            entanglement_type="linear",
            entanglement_freq=1,
        )
        assert not is_valid
        assert "qubits" in msg.lower()

    def test_validate_architecture_params_invalid_gate(self):
        is_valid, msg = validate_architecture_params(
            num_qubits=4,
            num_layers=2,
            rotation_gates=["invalid_gate"],
            entanglement_type="linear",
            entanglement_freq=1,
        )
        assert not is_valid
        assert "rotation_gates" in msg.lower()

    def test_validate_fitness_weights_valid(self):
        is_valid, msg = validate_fitness_weights(0.5, 0.3, 0.2)
        assert is_valid
        assert msg == ""

    def test_validate_fitness_weights_invalid_sum(self):
        is_valid, msg = validate_fitness_weights(0.8, 0.8, 0.8)
        assert not is_valid
        assert "sum" in msg.lower()

    def test_validate_fitness_weights_negative(self):
        is_valid, msg = validate_fitness_weights(-0.2, 0.7, 0.5)
        assert not is_valid
        assert "non-negative" in msg.lower()

    def test_validate_search_params_valid(self):
        is_valid, msg = validate_search_params(
            pop_size=10,
            generations=5,
            tournament_size=2,
        )
        assert is_valid
        assert msg == ""

    def test_validate_search_params_invalid(self):
        is_valid_pop, msg_pop = validate_search_params(
            pop_size=1,  # Too small
            generations=5,
            tournament_size=2,
        )
        assert not is_valid_pop
        assert "population" in msg_pop.lower()

        is_valid_gen, msg_gen = validate_search_params(
            pop_size=6,
            generations=0,  # Invalid
            tournament_size=2,
        )
        assert not is_valid_gen
        assert "generation" in msg_gen.lower()

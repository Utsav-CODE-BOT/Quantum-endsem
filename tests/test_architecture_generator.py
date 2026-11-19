"""Unit tests for ArchitectureSpace and ArchitectureGenerator."""

from __future__ import annotations

import pytest

from src.optimization.architecture_generator import ArchitectureGenerator
from src.optimization.architecture_space import (
    ArchitectureSpace,
    ArchitectureValidationError,
)
from src.quantum.architecture_schema import ArchitectureConfig
from src.utils.seed_manager import SeedManager


class TestArchitectureSpace:
    """Test validation of candidate configurations."""

    def test_valid_config_passes(self) -> None:
        space = ArchitectureSpace(qubit_choices=(4,))
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=2,
            rotation_gates=("ry", "rz"),
            entanglement_type="linear",
            entanglement_freq=1,
            feature_rotation="ry",
            measurement_strategy="parity",
        )
        assert space.validate(config) is True

    def test_invalid_qubits_raises(self) -> None:
        space = ArchitectureSpace(qubit_choices=(4,))
        config = ArchitectureConfig(
            num_qubits=5,  # Invalid
            num_layers=2,
            rotation_gates=("ry",),
            entanglement_type="linear",
        )
        with pytest.raises(ArchitectureValidationError, match="num_qubits"):
            space.validate(config)

    def test_invalid_entanglement_raises(self) -> None:
        space = ArchitectureSpace(qubit_choices=(4,))
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=2,
            rotation_gates=("ry",),
            entanglement_type="all_to_all",  # Unsupported
        )
        with pytest.raises(ArchitectureValidationError, match="Invalid entanglement_type"):
            space.validate(config)


class TestArchitectureGenerator:
    """Test sampling and population creation."""

    def test_sample_candidate(self) -> None:
        generator = ArchitectureGenerator(
            space=ArchitectureSpace(qubit_choices=(4,)),
            seed_manager=SeedManager(global_seed=42),
        )
        cand = generator.sample_candidate()
        assert isinstance(cand, ArchitectureConfig)
        assert cand.num_qubits == 4
        assert 1 <= cand.num_layers <= 5
        assert cand.arch_id.startswith("arch_")

    def test_sample_population_uniqueness(self) -> None:
        generator = ArchitectureGenerator(
            space=ArchitectureSpace(qubit_choices=(4,)),
            seed_manager=SeedManager(global_seed=42),
        )
        pop = generator.sample_population(size=8)
        assert len(pop) == 8
        arch_ids = [c.arch_id for c in pop]
        assert len(set(arch_ids)) == 8  # All unique

    def test_generator_reproducibility(self) -> None:
        gen1 = ArchitectureGenerator(
            space=ArchitectureSpace(qubit_choices=(4,)),
            seed_manager=SeedManager(global_seed=123),
        )
        pop1 = gen1.sample_population(size=5)

        gen2 = ArchitectureGenerator(
            space=ArchitectureSpace(qubit_choices=(4,)),
            seed_manager=SeedManager(global_seed=123),
        )
        pop2 = gen2.sample_population(size=5)

        assert [c.arch_id for c in pop1] == [c.arch_id for c in pop2]

    def test_compile_circuit(self) -> None:
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=2,
            rotation_gates=("ry", "rz"),
            entanglement_type="circular",
        )
        gen, meta = ArchitectureGenerator.compile_circuit(config)
        assert meta.num_qubits == 4
        assert meta.circuit_depth > 0
        assert meta.num_trainable_parameters == 24

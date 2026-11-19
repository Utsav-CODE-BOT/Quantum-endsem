"""Smoke tests for project scaffolding (Phase 1).

Validates that the project structure, configuration loading,
logging, seed management, and system monitoring work correctly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml


# ============================================================
# Test Group 1: Project Structure
# ============================================================

class TestProjectStructure:
    """Verify the project directory structure is correct."""

    def test_project_root_exists(self, project_root: Path) -> None:
        assert project_root.exists()
        assert project_root.is_dir()

    def test_configs_directory_exists(self, config_dir: Path) -> None:
        assert config_dir.exists()
        assert config_dir.is_dir()

    def test_all_config_files_exist(self, config_dir: Path) -> None:
        expected = [
            "dataset.yaml",
            "preprocessing.yaml",
            "quantum.yaml",
            "training.yaml",
            "evolutionary_search.yaml",
            "noise.yaml",
            "experiments.yaml",
        ]
        for filename in expected:
            filepath = config_dir / filename
            assert filepath.exists(), f"Missing config file: {filename}"

    def test_src_package_importable(self, project_root: Path) -> None:
        src_init = project_root / "src" / "__init__.py"
        assert src_init.exists()

    def test_all_subpackages_have_init(self, project_root: Path) -> None:
        subpackages = [
            "data", "preprocessing", "quantum", "training",
            "optimization", "evaluation", "baselines",
            "visualization", "experiments", "utils",
        ]
        for pkg in subpackages:
            init_file = project_root / "src" / pkg / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in src/{pkg}/"

    def test_data_directories_exist(self, project_root: Path) -> None:
        for subdir in ["raw", "processed", "metadata"]:
            assert (project_root / "data" / subdir).is_dir()

    def test_experiments_directories_exist(self, project_root: Path) -> None:
        for subdir in ["configs", "logs", "results", "checkpoints", "figures"]:
            assert (project_root / "experiments" / subdir).is_dir()


# ============================================================
# Test Group 2: Configuration Loading
# ============================================================

class TestConfigLoader:
    """Test the configuration management system."""

    def test_config_manager_creation(self, config_manager) -> None:
        assert config_manager is not None
        assert "dataset" in repr(config_manager)

    def test_load_dataset_config(self, config_manager) -> None:
        section = config_manager.get_section("dataset")
        assert "dataset" in section
        ds = section["dataset"]
        assert "selected_classes" in ds
        assert isinstance(ds["selected_classes"], list)
        assert len(ds["selected_classes"]) == 2

    def test_load_preprocessing_config(self, config_manager) -> None:
        section = config_manager.get_section("preprocessing")
        assert "preprocessing" in section
        pca_cfg = section["preprocessing"]["pca"]
        assert pca_cfg["n_components"] in [4, 6, 8]

    def test_load_quantum_config(self, config_manager) -> None:
        section = config_manager.get_section("quantum")
        assert "quantum" in section
        arch = section["quantum"]["default_architecture"]
        assert "n_qubits" in arch
        assert "n_layers" in arch

    def test_get_nested_value(self, config_manager) -> None:
        n_components = config_manager.get(
            "preprocessing", "preprocessing.pca.n_components"
        )
        assert n_components == 4

    def test_get_with_default(self, config_manager) -> None:
        val = config_manager.get("dataset", "nonexistent.key", default="fallback")
        assert val == "fallback"

    def test_override_value(self, config_manager) -> None:
        config_manager.override("preprocessing", "preprocessing.pca.n_components", 8)
        val = config_manager.get("preprocessing", "preprocessing.pca.n_components")
        assert val == 8

    def test_save_snapshot(self, config_manager, tmp_output_dir: Path) -> None:
        snapshot_path = tmp_output_dir / "config_snapshot.yaml"
        config_manager.save_snapshot(snapshot_path)
        assert snapshot_path.exists()

        # Verify it's valid YAML
        with open(snapshot_path) as f:
            loaded = yaml.safe_load(f)
        assert "dataset" in loaded

    def test_to_dict_returns_copy(self, config_manager) -> None:
        d1 = config_manager.to_dict()
        d2 = config_manager.to_dict()
        assert d1 == d2
        # Modify d1 and verify d2 is unaffected
        d1["dataset"]["modified"] = True
        d3 = config_manager.to_dict()
        assert "modified" not in d3.get("dataset", {})


# ============================================================
# Test Group 3: Logging
# ============================================================

class TestLogging:
    """Test the logging system."""

    def test_setup_logging(self) -> None:
        from src.utils.logging_config import setup_logging
        # Should not raise
        setup_logging(level="DEBUG", console=True)

    def test_get_logger(self) -> None:
        from src.utils.logging_config import get_logger
        logger = get_logger("test_module")
        assert logger is not None
        assert "qao" in logger.name

    def test_logger_writes_message(self, capfd) -> None:
        from src.utils.logging_config import get_logger, setup_logging
        setup_logging(level="DEBUG", console=True)
        logger = get_logger("test_logging")
        logger.info("Test message from Phase 1 smoke test")
        # Message should be written to stderr
        # (capfd captures stdout/stderr)

    def test_file_logging(self, tmp_output_dir: Path) -> None:
        from src.utils.logging_config import setup_logging, get_logger
        import importlib
        import src.utils.logging_config as lc

        # Reset initialized state for this test
        lc._initialized = False

        setup_logging(
            level="DEBUG",
            log_dir=tmp_output_dir,
            log_file="test_log.log",
            console=False,
        )
        logger = get_logger("test_file_log")
        logger.info("File log test message")

        log_file = tmp_output_dir / "test_log.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "File log test message" in content

        # Restore initialized state
        lc._initialized = False


# ============================================================
# Test Group 4: Seed Management
# ============================================================

class TestSeedManager:
    """Test the random seed management system."""

    def test_creation(self, seed_manager) -> None:
        assert seed_manager.global_seed == 12345

    def test_derive_seed_deterministic(self, seed_manager) -> None:
        s1 = seed_manager.derive_seed("component_a")
        s2 = seed_manager.derive_seed("component_a")
        assert s1 == s2  # Same component → same seed

    def test_derive_seed_unique(self, seed_manager) -> None:
        s1 = seed_manager.derive_seed("component_a")
        s2 = seed_manager.derive_seed("component_b")
        assert s1 != s2  # Different components → different seeds

    def test_set_all_seeds(self, seed_manager) -> None:
        import numpy as np
        seed_manager.set_all_seeds()
        # After setting seeds, numpy should produce deterministic values
        val1 = np.random.rand()
        seed_manager.set_all_seeds()
        val2 = np.random.rand()
        assert val1 == val2

    def test_get_numpy_rng(self, seed_manager) -> None:
        import numpy as np
        rng1 = seed_manager.get_numpy_rng("test_rng")
        val1 = rng1.random()
        rng2 = seed_manager.get_numpy_rng("test_rng")
        val2 = rng2.random()
        assert val1 == val2  # Same component → same RNG sequence

    def test_get_state(self, seed_manager) -> None:
        seed_manager.derive_seed("comp_x")
        state = seed_manager.get_state()
        assert state["global_seed"] == 12345
        assert "comp_x" in state["derived_seeds"]

    def test_different_global_seeds_produce_different_derived(self) -> None:
        from src.utils.seed_manager import SeedManager
        sm1 = SeedManager(global_seed=1)
        sm2 = SeedManager(global_seed=2)
        assert sm1.derive_seed("test") != sm2.derive_seed("test")


# ============================================================
# Test Group 5: System Monitor
# ============================================================

class TestSystemMonitor:
    """Test the system monitoring utilities."""

    def test_get_system_info(self) -> None:
        from src.utils.system_monitor import get_system_info
        info = get_system_info()
        assert info.python_version != ""
        assert info.cpu_cores_logical > 0
        assert info.ram_total_gb > 0

    def test_system_info_to_dict(self) -> None:
        from src.utils.system_monitor import get_system_info
        info = get_system_info()
        d = info.to_dict()
        assert isinstance(d, dict)
        assert "python_version" in d
        assert "ram_total_gb" in d

    def test_resource_snapshot(self) -> None:
        from src.utils.system_monitor import get_resource_snapshot
        snap = get_resource_snapshot()
        assert snap.timestamp != ""
        assert snap.ram_used_gb > 0
        assert 0 <= snap.ram_percent <= 100

    def test_timer(self) -> None:
        from src.utils.system_monitor import Timer
        import time
        with Timer("test_timer", log=False) as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04  # Allow small timing variance
        assert t.elapsed < 1.0

    def test_get_package_versions(self) -> None:
        from src.utils.system_monitor import get_package_versions
        versions = get_package_versions()
        assert isinstance(versions, dict)
        assert "numpy" in versions


# ============================================================
# Test Group 6: Config YAML Validation
# ============================================================

class TestConfigValidation:
    """Validate the content of configuration files."""

    def test_dataset_classes_valid(self, config_manager) -> None:
        classes = config_manager.get("dataset", "dataset.selected_classes")
        assert all(0 <= c <= 9 for c in classes)

    def test_pca_components_valid(self, config_manager) -> None:
        n = config_manager.get("preprocessing", "preprocessing.pca.n_components")
        assert n in [4, 6, 8]

    def test_quantum_qubits_valid(self, config_manager) -> None:
        n = config_manager.get("quantum", "quantum.default_architecture.n_qubits")
        assert 2 <= n <= 12

    def test_training_optimizer_valid(self, config_manager) -> None:
        opt = config_manager.get("training", "training.optimizer.name")
        assert opt in ["cobyla", "spsa", "l_bfgs_b"]

    def test_evolution_weights_sum_reasonable(self, config_manager) -> None:
        fitness = config_manager.get("evolutionary_search", "evolutionary_search.fitness")
        total = fitness["w_performance"] + fitness["w_complexity"] + fitness["w_noise_robustness"]
        assert 0.5 <= total <= 2.0  # Reasonable weight range

    def test_feature_scaling_range(self, config_manager) -> None:
        target = config_manager.get(
            "preprocessing", "preprocessing.feature_scaling.target_range"
        )
        assert len(target) == 2
        assert target[0] < target[1]
        assert target[0] >= 0

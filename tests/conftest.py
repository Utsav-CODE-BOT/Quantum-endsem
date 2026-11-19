"""Pytest configuration and shared fixtures.

Provides test fixtures for configuration, seed management,
and temporary directories used across all test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture
def config_manager():
    """Create a fresh ConfigManager instance for testing."""
    from src.utils.config_loader import ConfigManager
    return ConfigManager()


@pytest.fixture
def seed_manager():
    """Create a SeedManager with a fixed test seed."""
    from src.utils.seed_manager import SeedManager
    return SeedManager(global_seed=12345)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for test artifacts."""
    output = tmp_path / "test_output"
    output.mkdir(parents=True, exist_ok=True)
    return output

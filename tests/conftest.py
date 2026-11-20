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


def pytest_sessionfinish(session, exitstatus):
    """Write machine-readable test execution report upon test session completion."""
    import datetime
    import json
    import subprocess

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    passed = len(reporter.stats.get("passed", [])) if reporter else 0
    failed = len(reporter.stats.get("failed", [])) if reporter else 0
    skipped = len(reporter.stats.get("skipped", [])) if reporter else 0
    xfailed = len(reporter.stats.get("xfailed", [])) if reporter else 0
    xpassed = len(reporter.stats.get("xpassed", [])) if reporter else 0
    error = len(reporter.stats.get("error", [])) if reporter else 0
    total = passed + failed + skipped + xfailed + xpassed + error

    git_commit = "unknown"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            git_commit = res.stdout.strip()
    except Exception:
        pass

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "error": error,
        "exit_code": int(exitstatus),
        "status": "passed" if int(exitstatus) == 0 and failed == 0 and error == 0 else "failed",
    }

    report_path = PROJECT_ROOT / "experiments" / "results" / "test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

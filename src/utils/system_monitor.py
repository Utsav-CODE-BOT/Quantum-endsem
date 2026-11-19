"""System resource monitoring utilities.

Tracks CPU, memory, and timing for experiment metadata.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SystemInfo:
    """Snapshot of system information for experiment reproducibility."""
    os_name: str = ""
    os_version: str = ""
    platform: str = ""
    python_version: str = ""
    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    ram_total_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_name": self.cpu_name,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "ram_total_gb": self.ram_total_gb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
        }


def get_system_info() -> SystemInfo:
    """Collect system information.

    Returns:
        SystemInfo dataclass with hardware and software details.
    """
    info = SystemInfo()
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.platform = platform.platform()
    info.python_version = sys.version
    info.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
    info.cpu_cores_logical = psutil.cpu_count(logical=True) or 0

    mem = psutil.virtual_memory()
    info.ram_total_gb = round(mem.total / (1024 ** 3), 2)

    # CPU name (platform-dependent)
    try:
        info.cpu_name = platform.processor() or "Unknown"
    except Exception:
        info.cpu_name = "Unknown"

    # GPU check
    try:
        import torch
        info.gpu_available = torch.cuda.is_available()
        if info.gpu_available:
            info.gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        info.gpu_available = False

    return info


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage snapshot."""
    timestamp: str = ""
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_available_gb": self.ram_available_gb,
            "ram_percent": self.ram_percent,
        }


def get_resource_snapshot() -> ResourceSnapshot:
    """Take a snapshot of current resource usage."""
    mem = psutil.virtual_memory()
    return ResourceSnapshot(
        timestamp=datetime.now().isoformat(),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        ram_used_gb=round(mem.used / (1024 ** 3), 2),
        ram_available_gb=round(mem.available / (1024 ** 3), 2),
        ram_percent=mem.percent,
    )


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("VQC Training") as t:
            train_model()
        print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __init__(self, label: str = "", log: bool = True) -> None:
        self.label = label
        self.log = log
        self._start: float = 0.0
        self._end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        if self.log and self.label:
            logger.info("[Timer] %s started", self.label)
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start
        if self.log and self.label:
            logger.info("[Timer] %s completed in %.2f seconds", self.label, self.elapsed)


def get_package_versions() -> dict[str, str]:
    """Get versions of key installed packages for experiment metadata."""
    packages = {}
    for pkg_name in [
        "qiskit", "qiskit_aer", "qiskit_machine_learning", "qiskit_algorithms",
        "numpy", "scipy", "sklearn", "pandas", "matplotlib", "torch", "psutil",
    ]:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
            # Normalize package name for display
            display_name = pkg_name.replace("_", "-")
            if display_name == "sklearn":
                display_name = "scikit-learn"
            packages[display_name] = str(version)
        except ImportError:
            pass
    return packages

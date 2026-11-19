"""Benchmark comparison service loading classical and quantum model evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config_loader import get_project_root


class BenchmarkService:
    """Service providing access to standardized comparative benchmark records."""

    @staticmethod
    def load_benchmark_data() -> dict[str, Any]:
        """Load benchmark comparison data from disk."""
        path = get_project_root() / "experiments" / "results" / "benchmark_comparison.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_records(cls) -> list[dict[str, Any]]:
        """Return list of evaluated model benchmark dictionaries."""
        data = cls.load_benchmark_data()
        return data.get("results", [])

    @classmethod
    def get_model_by_name(cls, name: str) -> dict[str, Any] | None:
        """Find benchmark record by model name."""
        for r in cls.get_records():
            if r.get("model_name") == name:
                return r
        return None

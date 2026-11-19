"""Formatting utilities for scientific metrics and time measurements."""

from __future__ import annotations

from typing import Any


def format_percentage(val: float | None, decimals: int = 2) -> str:
    """Format floating point score (e.g. 0.9545) as percentage string (95.45%)."""
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_time(seconds: float | None) -> str:
    """Format seconds into readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    mins = int(seconds // 60)
    rem_sec = seconds % 60
    return f"{mins}m {rem_sec:.1f}s"


def format_latency(ms: float | None) -> str:
    """Format latency in milliseconds."""
    if ms is None:
        return "N/A"
    if ms < 0.1:
        return f"{ms * 1000:.1f} µs/sample"
    return f"{ms:.2f} ms/sample"


def format_parameters(num: int | None) -> str:
    """Format parameter count."""
    if num is None:
        return "N/A"
    return f"{num:,}"

"""Session state management utilities for Streamlit dashboard."""

from __future__ import annotations

from typing import Any
import streamlit as st


def init_session_state() -> None:
    """Initialize default session state keys if not already present."""
    defaults: dict[str, Any] = {
        "active_experiment_id": "exp_baseline_vqc",
        "search_running": False,
        "search_progress": {
            "status": "idle",
            "current_gen": 0,
            "total_gen": 0,
            "best_fitness": 0.0,
            "best_val_acc": 0.0,
            "elapsed_time": 0.0,
            "history": [],
            "error": None,
        },
        "selected_architecture": None,
        "selected_compare_models": [],
        "filter_class": "All",
        "current_sample_idx": 0,
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_state(key: str, default: Any = None) -> Any:
    """Get value from Streamlit session state safely."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set value in Streamlit session state."""
    st.session_state[key] = value

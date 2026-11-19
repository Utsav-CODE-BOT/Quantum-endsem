"""Technical KPI metric card rendering components."""

from __future__ import annotations

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
) -> None:
    """Render a single dark-themed research KPI card.

    Args:
        title: Card header (e.g. "Best Test Accuracy").
        value: Main numeric/string metric value (e.g. "96.00%").
        delta: Optional secondary label or delta comparison (e.g. "+1.00% vs baseline").
        delta_type: 'positive' (green), 'negative' (red), or 'neutral' (slate).
    """
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-card-delta {delta_type}">{delta}</div>'

    html = f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_grid_row(cards: list[dict[str, str]]) -> None:
    """Render a row of metric cards evenly spaced across columns.

    Args:
        cards: List of dicts with keys 'title', 'value', 'delta' (optional), 'delta_type' (optional).
    """
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            render_metric_card(
                title=card.get("title", ""),
                value=card.get("value", "N/A"),
                delta=card.get("delta"),
                delta_type=card.get("delta_type", "neutral"),
            )

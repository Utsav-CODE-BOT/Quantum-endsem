"""Sidebar component for Quantum Architecture Optimizer dashboard."""

from __future__ import annotations

import streamlit as st
from ui.services.search_service import SearchService


def render_sidebar() -> None:
    """Render unified navigation and status sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; color: #818cf8; font-size: 1.3rem; font-weight: 800; letter-spacing: -0.02em;">
                    ⚛️ QUANTUM ARCH
                </h2>
                <div style="font-size: 0.76rem; color: #94a3b8; font-weight: 500; margin-top: 0.2rem;">
                    Resource-Aware VQC Optimizer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Backend Status Indicator
        is_searching = SearchService.is_running()
        if is_searching:
            status_html = '<span class="status-badge active">● SEARCH IN PROGRESS</span>'
        else:
            status_html = '<span class="status-badge ready">● BACKEND READY</span>'

        st.markdown(
            f"""
            <div style="background: #131c31; border: 1px solid #1e293b; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 1.2rem;">
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase;">System Engine</div>
                <div style="margin-top: 0.35rem;">{status_html}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.4rem; font-family: monospace;">
                    Qiskit 1.4.2 · Aer 0.16.4
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">
                Research Navigation
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Guidance footnote
        st.markdown(
            """
            <div style="margin-top: 2rem; font-size: 0.72rem; color: #64748b; line-height: 1.4;">
                <strong>Dataset:</strong> Fashion-MNIST (0 vs 9)<br>
                <strong>Method:</strong> Zero-Leakage PCA & Angle Encoding<br>
                <strong>Simulation:</strong> Aer Density Matrix
            </div>
            """,
            unsafe_allow_html=True,
        )

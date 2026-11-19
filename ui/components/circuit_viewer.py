"""Quantum circuit visualizer and export component for Streamlit."""

from __future__ import annotations

import streamlit as st
from qiskit import QuantumCircuit

from src.quantum.circuit_generator import CircuitMetadata
from ui.services.circuit_service import CircuitService


def render_circuit_component(
    circuit: QuantumCircuit,
    meta: CircuitMetadata | None = None,
    key_prefix: str = "circ",
) -> None:
    """Render quantum circuit diagram with Matplotlib/Text toggles and export controls.

    Args:
        circuit: Qiskit QuantumCircuit instance.
        meta: Optional CircuitMetadata dataclass.
        key_prefix: Unique key prefix for widgets.
    """
    if meta is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Qubits", meta.num_qubits)
        with c2:
            st.metric("Circuit Depth", meta.circuit_depth)
        with c3:
            st.metric("Total Gates", meta.total_gates)
        with c4:
            st.metric("2-Qubit (CX) Gates", meta.two_qubit_gates)
        with c5:
            st.metric("Trainable Parameters", meta.num_trainable_parameters)

    view_mode = st.radio(
        "Circuit Visualization Mode",
        options=["Graphical (Matplotlib)", "Text (Unicode ASCII)"],
        horizontal=True,
        key=f"{key_prefix}_view_mode",
    )

    if view_mode == "Graphical (Matplotlib)":
        try:
            fig = CircuitService.draw_matplotlib(circuit)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Matplotlib rendering fallback to text: {e}")
            text_diag = CircuitService.draw_text(circuit)
            st.code(text_diag, language="text")
    else:
        text_diag = CircuitService.draw_text(circuit)
        st.code(text_diag, language="text")

    # Download controls
    try:
        png_bytes = CircuitService.export_circuit_png_bytes(circuit)
        st.download_button(
            label="⬇️ Download Circuit Diagram (PNG, 200 DPI)",
            data=png_bytes,
            file_name=f"{circuit.name or 'quantum_circuit'}.png",
            mime="image/png",
            key=f"{key_prefix}_download_png",
        )
    except Exception as e:
        st.caption(f"Circuit image export unavailable: {e}")

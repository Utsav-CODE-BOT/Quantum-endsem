"""Page 3: Quantum Circuit Lab — Interactive VQC Architecture Builder and Visualizer."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.quantum.architecture_schema import ArchitectureConfig
from ui.components.circuit_viewer import render_circuit_component
from ui.components.sidebar import render_sidebar
from ui.services.circuit_service import CircuitService
from ui.utils.state import init_session_state
from ui.utils.validation import validate_architecture_params

st.set_page_config(
    page_title="Quantum Circuit Lab | Quantum Architecture Optimizer",
    page_icon="⚡",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("⚡ Quantum Circuit Lab")
st.markdown(
    "Design, compile, and analyze **Variational Quantum Classifier (VQC)** circuits. "
    "All circuits are compiled in real time using the project's native Qiskit backend, extracting exact "
    "decomposed depth, gate distributions, and trainable parameter counts."
)

st.markdown("---")

# Canonical Architectures Quick-Loader
st.subheader("📚 Quick-Load Canonical Architectures")
canon_archs = CircuitService.get_canonical_architectures()
cols_btn = st.columns(len(canon_archs))

selected_preset = None
for col, (name, cfg) in zip(cols_btn, canon_archs.items()):
    with col:
        if st.button(f"Load {name}", key=f"btn_{name}"):
            selected_preset = cfg
            st.session_state["selected_architecture"] = cfg

# If a preset was selected, load its values into defaults
current_cfg = st.session_state.get("selected_architecture") or canon_archs["Search-Optimized (arch_6327d7bcb8)"]

st.markdown("---")
st.subheader("🛠️ Architecture Parameter Configuration")

col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

with col_cfg1:
    qubits = st.selectbox(
        "Qubits (Feature Dimension)",
        options=[2, 4, 6, 8],
        index=[2, 4, 6, 8].index(current_cfg.num_qubits) if current_cfg.num_qubits in [2, 4, 6, 8] else 1,
        help="Number of qubits in the circuit. Must match input PCA dimensionality.",
    )
    layers = st.slider(
        "Variational Layers (Repetitions)",
        min_value=1,
        max_value=6,
        value=current_cfg.num_layers,
        help="Number of repeating variational ansatz layers.",
    )

with col_cfg2:
    rotations = st.multiselect(
        "Ansatz Rotation Gates",
        options=["rx", "ry", "rz"],
        default=list(current_cfg.rotation_gates) if current_cfg.rotation_gates else ["ry"],
        help="Single-qubit parameterized rotation gates applied per qubit per layer.",
    )
    feat_rot = st.selectbox(
        "Feature Encoding Gate",
        options=["ry", "rx", "rz"],
        index=["ry", "rx", "rz"].index(current_cfg.feature_rotation) if current_cfg.feature_rotation in ["ry", "rx", "rz"] else 0,
        help="Gate used to encode normalized input features into rotation angles.",
    )

with col_cfg3:
    ent_type = st.selectbox(
        "Entanglement Topology",
        options=["linear", "circular", "pairwise", "none"],
        index=["linear", "circular", "pairwise", "none"].index(current_cfg.entanglement_type) if current_cfg.entanglement_type in ["linear", "circular", "pairwise", "none"] else 0,
        help="Two-qubit CNOT coupling pattern across qubits.",
    )
    ent_freq = st.slider(
        "Entanglement Frequency",
        min_value=1,
        max_value=3,
        value=current_cfg.entanglement_freq,
        help="Apply entangling CNOT gates every k layers (1 = every layer).",
    )

# Validate parameters
is_valid, err_msg = validate_architecture_params(qubits, layers, rotations, ent_type, ent_freq)

if not is_valid:
    st.error(f"Configuration Error: {err_msg}")
else:
    # Build configuration object
    arch_config = ArchitectureConfig(
        num_qubits=qubits,
        num_layers=layers,
        rotation_gates=tuple(rotations),
        entanglement_type=ent_type,
        entanglement_freq=ent_freq,
        feature_rotation=feat_rot,
        measurement_strategy="parity",
    )

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1f293d; border-radius: 6px; padding: 0.6rem 1rem; margin: 0.8rem 0;">
            <span style="color: #94a3b8; font-size: 0.8rem;">Deterministic Architecture ID:</span>
            <code style="color: #38bdf8; font-size: 0.95rem; margin-left: 0.5rem; font-weight: 700;">{arch_config.arch_id}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compile circuit in real-time
    with st.spinner("Compiling Qiskit quantum circuit..."):
        circuit, metadata = CircuitService.compile_from_config(arch_config)

    st.subheader("📐 Compiled Quantum Circuit Diagram")
    render_circuit_component(circuit, metadata, key_prefix="lab")

    # Detailed Gate Breakdown Table
    with st.expander("🔍 Detailed Gate Distribution Breakdown"):
        st.json(metadata.to_dict())

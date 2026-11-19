"""Main Application Entry Point: Quantum Architecture Optimizer Research Platform."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui.components.metric_cards import render_metric_card
from ui.components.sidebar import render_sidebar
from ui.services.benchmark_service import BenchmarkService
from ui.services.dataset_service import DatasetService
from ui.services.experiment_service import ExperimentService
from ui.services.noise_service import NoiseService
from ui.services.search_service import SearchService
from ui.utils.formatting import format_percentage
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Quantum Architecture Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark stylesheet
css_path = Path(__file__).resolve().parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize global state and render sidebar
init_session_state()
render_sidebar()

# Hero Header
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #131c31 0%, #0d1527 100%); border: 1px solid #1e293b; border-radius: 12px; padding: 2rem 2.5rem; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
            <span style="font-size: 2.8rem;">⚛️</span>
            <div>
                <h1 style="color: #f8fafc; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;">
                    Quantum Architecture Optimizer
                </h1>
                <div style="font-size: 1.15rem; color: #38bdf8; font-weight: 600;">
                    Self-Adaptive Quantum Neural Networks with Resource-Aware Architecture Optimization
                </div>
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 1rem; line-height: 1.6; max-width: 900px; margin-top: 1rem; margin-bottom: 0;">
            An interactive research platform investigating the multi-objective trade-offs between classification performance,
            quantum circuit complexity (depth and two-qubit gate counts), and robustness under simulated NISQ quantum noise
            on the Fashion-MNIST dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# System Health & Research Highlights (Authoritative Dynamic Loading)
best_arch = SearchService.load_saved_best_architecture()
noise_data = NoiseService.load_saved_results()
base_path = ExperimentService.get_results_dir() / "baseline_vqc_results.json"
base_exp = ExperimentService.load_raw_experiment_file(str(base_path)) if base_path.exists() else {}

# 1. Test Accuracy
opt_test_acc = best_arch.get("test_metrics", {}).get("accuracy")
base_test_acc = base_exp.get("test_metrics", {}).get("accuracy")
if opt_test_acc is not None:
    val_acc_str = format_percentage(opt_test_acc)
    delta_acc = f"Baseline: {format_percentage(base_test_acc)}" if base_test_acc is not None else "Discovered"
else:
    val_acc_str = "N/A"
    delta_acc = "Run pipeline to populate"

# 2. Circuit Depth Reduction
opt_depth = best_arch.get("circuit_metadata", {}).get("circuit_depth")
base_depth = base_exp.get("circuit_metadata", {}).get("circuit_depth")
if opt_depth is not None and base_depth is not None and base_depth > 0:
    red_pct = ((base_depth - opt_depth) / base_depth) * 100
    val_depth_str = f"-{red_pct:.1f}%"
    delta_depth = f"{base_depth} → {opt_depth} circuit depth"
elif opt_depth is not None:
    val_depth_str = f"Depth {opt_depth}"
    delta_depth = "Optimized candidate"
else:
    val_depth_str = "N/A"
    delta_depth = "Run pipeline to populate"

# 3. Two-Qubit CX/CZ Gates
opt_2q = best_arch.get("circuit_metadata", {}).get("two_qubit_gates")
if opt_2q is not None:
    val_2q_str = f"{opt_2q} Gates"
    delta_2q = "Zero 2-qubit noise penalty" if opt_2q == 0 else f"{opt_2q} entangling gates"
else:
    val_2q_str = "N/A"
    delta_2q = "Run pipeline to populate"

# 4. Noise Robustness (High Noise Retention)
high_noise_retention = None
if "benchmarks" in noise_data and "optimized" in noise_data["benchmarks"]:
    for b in noise_data["benchmarks"]["optimized"]:
        if b.get("noise_config", {}).get("name") == "High_Noise":
            high_noise_retention = b.get("relative_accuracy_retention")
            break

if high_noise_retention is not None:
    val_noise_str = format_percentage(high_noise_retention)
    delta_noise = "Retention under High NISQ Noise"
else:
    val_noise_str = "N/A"
    delta_noise = "No noise benchmark run"

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Optimized Test Acc",
        value=val_acc_str,
        delta=delta_acc,
        icon="🎯",
    )
with col2:
    render_metric_card(
        title="Circuit Depth Reduction",
        value=val_depth_str,
        delta=delta_depth,
        icon="📉",
    )
with col3:
    render_metric_card(
        title="CX Entanglement Gates",
        value=val_2q_str,
        delta=delta_2q,
        icon="🛡️",
    )
with col4:
    render_metric_card(
        title="NISQ Noise Robustness",
        value=val_noise_str,
        delta=delta_noise,
        icon="⚡",
    )

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# Module Navigation Grid
st.markdown(
    """
    <h3 style="color: #f8fafc; font-weight: 700; margin-bottom: 1rem;">
        🧭 Research Modules & Capabilities
    </h3>
    """,
    unsafe_allow_html=True,
)

modules = [
    {
        "title": "1. Overview Dashboard",
        "icon": "📊",
        "page": "pages/1_Overview.py",
        "desc": "Executive dashboard featuring key research performance indicators, convergence curves, and comparative charts.",
        "badge": "Core Dashboard",
    },
    {
        "title": "2. Dataset Explorer",
        "icon": "🖼️",
        "page": "pages/2_Dataset_Explorer.py",
        "desc": "Inspect Fashion-MNIST samples, PCA dimensionality reduction, angle encoding transformations, and feature variance.",
        "badge": "Zero Data Leakage",
    },
    {
        "title": "3. Quantum Circuit Lab",
        "icon": "🔬",
        "page": "pages/3_Quantum_Circuit_Lab.py",
        "desc": "Interactively design parameterized quantum circuits, compile with Qiskit, and export publication-ready circuit diagrams.",
        "badge": "Real-time Qiskit",
    },
    {
        "title": "4. Architecture Search",
        "icon": "🧬",
        "page": "pages/4_Architecture_Search.py",
        "desc": "Launch or inspect evolutionary multi-objective architecture optimization with real-time fitness telemetry.",
        "badge": "Multi-Objective",
    },
    {
        "title": "5. Noise Analysis",
        "icon": "⚡",
        "page": "pages/5_Noise_Analysis.py",
        "desc": "Simulate NISQ noise models (depolarizing, thermal relaxation, readout error) using Qiskit Aer density matrix simulation.",
        "badge": "Qiskit Aer",
    },
    {
        "title": "6. Model Benchmarking",
        "icon": "⚖️",
        "page": "pages/6_Model_Benchmarking.py",
        "desc": "Rigorous benchmarking against 7 classical and quantum baselines (SVM, MLP, Shallow, Standard, Deep VQCs).",
        "badge": "Fair Comparison",
    },
    {
        "title": "7. Experiment History",
        "icon": "📜",
        "page": "pages/7_Experiment_History.py",
        "desc": "Search, filter, and inspect persisted experiment runs and metrics, with complete JSON viewing and CSV export.",
        "badge": "Audit Trail",
    },
    {
        "title": "8. Documentation",
        "icon": "📚",
        "page": "pages/8_Documentation.py",
        "desc": "Explore complete system architecture, mathematical formulations, experimental protocol, and research reports.",
        "badge": "Knowledge Base",
    },
]

# Display 4 cards per row
for r in range(0, len(modules), 2):
    row_cols = st.columns(2)
    for c in range(2):
        if r + c < len(modules):
            mod = modules[r + c]
            with row_cols[c]:
                st.markdown(
                    f"""
                    <div style="background-color: #131c31; border: 1px solid #1e293b; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; min-height: 140px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                                {mod['icon']} {mod['title']}
                            </span>
                            <span style="background-color: rgba(6, 182, 212, 0.15); color: #06b6d4; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(6, 182, 212, 0.3);">
                                {mod['badge']}
                            </span>
                        </div>
                        <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                            {mod['desc']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {mod['title']}", key=f"btn_{mod['page']}", use_container_width=True):
                    st.switch_page(mod['page'])

st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

# Footer info
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; border-top: 1px solid #1e293b; padding-top: 1.5rem; margin-top: 2rem;">
        Quantum Architecture Optimizer &bull; Phase 13 Interactive Research Platform &bull; Built with Streamlit, Plotly & Qiskit
    </div>
    """,
    unsafe_allow_html=True,
)

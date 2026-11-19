"""Page 6: Model Benchmarking — Comparative Analysis of Classical and Quantum Models."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components.charts import create_model_comparison_bar, create_pareto_scatter
from ui.components.sidebar import render_sidebar
from ui.services.benchmark_service import BenchmarkService
from ui.utils.formatting import format_latency, format_parameters, format_percentage, format_time
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Model Benchmarking | Quantum Architecture Optimizer",
    page_icon="⚖️",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("⚖️ Classical & Quantum Model Benchmarking")
st.markdown(
    "Standardized side-by-side performance evaluation comparing classical machine learning baselines "
    "(Linear SVM, RBF SVM, MLP) against canonical fixed quantum circuits and the evolutionary search-optimized architecture."
)

st.markdown(
    """
    <div class="warning-callout">
        <strong>Scientific Rigor & Fair Comparison Statement:</strong> All models were evaluated on the identical 4-dimensional PCA feature
        representations of Fashion-MNIST using strict zero-leakage partitions. Classical SVMs solve this subproblem in milliseconds
        with 100% accuracy. The genuine research contribution of this project is <em>not</em> claiming quantum supremacy on toy datasets,
        but demonstrating automated quantum circuit resource reduction under physical NISQ constraints.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Load benchmark records
records = BenchmarkService.get_records()

if not records:
    st.warning("No benchmark comparison records found. Run `scripts/run_benchmarks.py` to generate results.")
else:
    # 1. Comparison Table
    st.subheader("📋 Standardized Model Comparison Table")

    table_data = []
    for r in records:
        table_data.append({
            "Model Name": r.get("model_name"),
            "Category": r.get("category"),
            "Parameters": format_parameters(r.get("num_parameters")),
            "Depth": r.get("circuit_depth") if r.get("circuit_depth") is not None else "—",
            "2Q (CX) Gates": r.get("two_qubit_gates") if r.get("two_qubit_gates") is not None else "—",
            "Val Acc": format_percentage(r.get("val_accuracy")),
            "Test Acc": format_percentage(r.get("test_accuracy")),
            "Test F1": format_percentage(r.get("test_f1")),
            "ROC-AUC": f"{r.get('test_roc_auc', 0.0):.4f}",
            "Train Time": format_time(r.get("training_time_seconds")),
            "Latency / Sample": format_latency(r.get("latency_ms_per_sample")),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Comparative Visualizations
    st.subheader("📊 Comparative Visualizations")

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        fig_bar = create_model_comparison_bar(records)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_chart2:
        # Parameter & Depth Comparison for Quantum Models
        q_records = [r for r in records if r.get("circuit_depth") is not None]
        if q_records:
            fig_comp = go.Figure()
            q_names = [r["model_name"] for r in q_records]
            depths = [r["circuit_depth"] for r in q_records]
            params = [r["num_parameters"] for r in q_records]

            fig_comp.add_trace(go.Bar(x=q_names, y=depths, name="Circuit Depth", marker={"color": "#f59e0b"}))
            fig_comp.add_trace(go.Bar(x=q_names, y=params, name="Trainable Parameters", marker={"color": "#8b5cf6"}))

            fig_comp.update_layout(
                title="<b>Quantum Resource Comparison: Depth vs Parameters</b>",
                paper_bgcolor="#0e1526",
                plot_bgcolor="#131c31",
                font={"color": "#94a3b8"},
                barmode="group",
                legend={"bgcolor": "rgba(19, 28, 49, 0.8)", "font": {"color": "#f8fafc"}},
                xaxis={"gridcolor": "#1e293b"},
                yaxis={"gridcolor": "#1e293b", "title": "Resource Count"},
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # 3. Depth Degradation & Trainability Analysis
    st.markdown("---")
    st.subheader("📉 Circuit Depth vs Trainability Trade-off")

    st.markdown(
        """
        Observing the empirical performance across varying quantum circuit depths illustrates the depth-dependent trainability bottleneck:
        """
    )
    q_models = [r for r in records if "VQC" in r.get("model_name", "")]
    for qm in q_models:
        name = qm.get("model_name", "VQC")
        depth = qm.get("circuit_depth", "N/A")
        params = qm.get("parameters", "N/A")
        acc = format_percentage(qm.get("test_accuracy")) if qm.get("test_accuracy") is not None else "N/A"
        runtime = format_time(qm.get("training_time_seconds")) if qm.get("training_time_seconds") is not None else "N/A"
        st.markdown(f"- **{name} (Depth {depth}, {params} params):** Reached **{acc} accuracy** in {runtime}.")

    st.info(
        "**Theoretical Note on Trainability & Barren Plateaus:** "
        "The empirical performance drop observed at deeper ansatz layers is consistent with trainability bottlenecks "
        "and overparameterization in variational circuits. In general variational quantum circuits, deep unstructured ansätze "
        "exhibit exponentially vanishing gradient variance (barren plateaus), $\\mathrm{Var}_\\theta[\\partial \\mathcal{L}/\\partial \\theta_k] \\in \\mathcal{O}(2^{-n})$. "
        "However, for this 4-qubit configuration, formal proof of barren plateaus requires explicit gradient variance scaling measurements across system size $n$."
    )

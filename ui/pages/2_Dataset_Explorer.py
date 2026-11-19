"""Page 2: Fashion-MNIST Dataset Explorer and Preprocessing Pipeline Inspection."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.components.charts import create_class_distribution_chart, create_pca_variance_plot
from ui.components.sidebar import render_sidebar
from ui.services.dataset_service import DatasetService
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Dataset Explorer | Quantum Architecture Optimizer",
    page_icon="🖼️",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("🖼️ Fashion-MNIST Dataset & Preprocessing Explorer")
st.markdown(
    "Inspect the raw Fashion-MNIST benchmark images, binary class partitions, and the step-by-step "
    "**zero-leakage** PCA dimensionality reduction and angle scaling pipeline."
)

# Load dataset summary
summary = DatasetService.get_dataset_summary()

# Summary Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Preprocessed Samples", f"{summary['total_preprocessed']:,}")
with c2:
    st.metric("Train Split", f"{summary['train_samples']} samples (balanced)")
with c3:
    st.metric("Validation / Test", f"{summary['val_samples']} val / {summary['test_samples']} test")
with c4:
    st.metric("Quantum Features (PCA)", f"{summary['num_pca_components']} components ({summary['cumulative_variance']*100:.1f}% var)")

st.markdown("---")

# Section 1: Preprocessing Pipeline Walkthrough
st.subheader("🔬 Step-by-Step Quantum Preprocessing Pipeline")

st.markdown(
    """
    <div class="research-callout">
        <strong>Strict Zero-Leakage Guarantee:</strong> Principal Component Analysis (PCA) and min/max feature scalers are
        fitted exclusively on the training partition. Validation and test samples are transformed using the pre-fitted parameters,
        preventing any data leakage into model training or evolutionary search.
    </div>
    """,
    unsafe_allow_html=True,
)

# Select sample to inspect
col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    split_choice = st.selectbox("Select Split to Sample From", ["train", "test"])
    sample_index = st.slider("Select Sample Index", min_value=0, max_value=199, value=0, step=1)
    if st.button("🎲 Random Sample"):
        sample_index = int(np.random.randint(0, 150))

with col_ctrl2:
    insp = DatasetService.inspect_sample_pipeline(sample_idx=sample_index, split=split_choice)
    st.markdown(
        f"**Sample #{sample_index}** — Ground Truth Class: **{insp['class_name']}** (Label: `{insp['original_label']}`)"
    )

# 4-stage pipeline visualization
stage1, stage2, stage3, stage4 = st.columns(4)

with stage1:
    st.markdown("**1. Raw Image (28×28)**")
    fig_raw, ax_raw = plt.subplots(figsize=(2.5, 2.5), dpi=100)
    ax_raw.imshow(insp["raw_image"], cmap="bone")
    ax_raw.axis("off")
    fig_raw.patch.set_facecolor("#131c31")
    st.pyplot(fig_raw, use_container_width=True)
    st.caption("Pixel intensities: [0, 255]")

with stage2:
    st.markdown("**2. Normalized & Flattened**")
    norm_vec = insp["normalized_flat"]
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(y=norm_vec, mode="lines", line={"color": "#38bdf8", "width": 1}))
    fig_norm.update_layout(
        paper_bgcolor="#131c31",
        plot_bgcolor="#0e1526",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=180,
        xaxis={"showticklabels": False},
        yaxis={"range": [0, 1]},
    )
    st.plotly_chart(fig_norm, use_container_width=True)
    st.caption("784-dimensional vector normalized to [0, 1]")

with stage3:
    st.markdown("**3. PCA Features (k=4)**")
    pca_vals = insp["pca_features"]
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(4)], y=pca_vals, marker={"color": "#818cf8"}))
    fig_pca.update_layout(
        paper_bgcolor="#131c31",
        plot_bgcolor="#0e1526",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=180,
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("Top 4 principal components")

with stage4:
    st.markdown("**4. Quantum Angles [0, π]**")
    angles = insp["scaled_angles"]
    fig_ang = go.Figure()
    fig_ang.add_trace(go.Bar(
        x=[f"q{i}" for i in range(4)],
        y=angles,
        marker={"color": "#34d399"},
        text=[f"{a:.2f} rad" for a in angles],
        textposition="auto",
    ))
    fig_ang.update_layout(
        paper_bgcolor="#131c31",
        plot_bgcolor="#0e1526",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=180,
        yaxis={"range": [0, np.pi + 0.2]},
    )
    st.plotly_chart(fig_ang, use_container_width=True)
    st.caption("Rotations: $R_y(\\theta_i)$ on qubits 0..3")

st.markdown("---")

# Section 2: PCA Variance Analysis & Class Distributions
col_dist1, col_dist2 = st.columns(2)

with col_dist1:
    st.subheader("📊 Class Balance")
    data = DatasetService.get_quantum_ready_data()
    y_train = data["y_train"]
    counts = {
        "Class 0 (T-shirt/top)": int(np.sum(y_train == 0)),
        "Class 9 (Ankle boot)": int(np.sum(y_train == 1)),
    }
    fig_class = create_class_distribution_chart(counts)
    st.plotly_chart(fig_class, use_container_width=True)

with col_dist2:
    st.subheader("📈 PCA Explained Variance")
    explained_ratios = summary.get("explained_variance_ratio", [])
    if explained_ratios:
        fig_var = create_pca_variance_plot(explained_ratios)
        st.plotly_chart(fig_var, use_container_width=True)

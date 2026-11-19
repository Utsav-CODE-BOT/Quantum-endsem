"""Page 7: Experiment History — Archive, Comparison, and Artifact Export."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.services.experiment_service import ExperimentService
from ui.utils.formatting import format_percentage, format_time
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Experiment History | Quantum Architecture Optimizer",
    page_icon="📁",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("📁 Research Experiment Archive")
st.markdown(
    "Inspect, filter, compare, and export historically persisted experimental results, "
    "model configurations, and quantum circuit evaluation logs."
)

st.markdown("---")

experiments = ExperimentService.list_saved_experiments()

if not experiments:
    st.info("No saved experiment results found in `experiments/results/`.")
else:
    # 1. Filters & Search Controls
    col_filter1, col_filter2, col_export = st.columns([1.5, 1.5, 1])

    with col_filter1:
        categories = ["All"] + sorted(list({e["category"] for e in experiments}))
        selected_cat = st.selectbox("Filter by Category", categories)

    with col_filter2:
        search_query = st.text_input("Search by ID or Title", placeholder="e.g. baseline or search")

    # Apply filters
    filtered_exps = experiments
    if selected_cat != "All":
        filtered_exps = [e for e in filtered_exps if e["category"] == selected_cat]
    if search_query:
        q = search_query.lower()
        filtered_exps = [e for e in filtered_exps if q in e["experiment_id"].lower() or q in e["title"].lower()]

    with col_export:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        csv_data = ExperimentService.export_records_to_csv(filtered_exps)
        st.download_button(
            label="⬇️ Export Table (CSV)",
            data=csv_data,
            file_name="quantum_experiments_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 2. Summary Table
    st.subheader(f"📋 Persisted Runs ({len(filtered_exps)} of {len(experiments)})")

    table_data = []
    for e in filtered_exps:
        table_data.append({
            "Experiment ID": e["experiment_id"],
            "Title": e["title"],
            "Category": e["category"],
            "Timestamp": e["timestamp"],
            "Val Acc": format_percentage(e["val_accuracy"]),
            "Test Acc": format_percentage(e["test_accuracy"]),
            "Depth": e["circuit_depth"] if e["circuit_depth"] is not None else "—",
            "Parameters": e["parameters"] if e["parameters"] is not None else "—",
            "Runtime": format_time(e["runtime_seconds"]),
            "Status": e["status"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3. Experiment Detail Inspector
    st.subheader("🔍 Experiment Payload Inspector")

    exp_options = {e["experiment_id"]: e for e in filtered_exps}
    selected_id = st.selectbox("Select Experiment to Inspect Details", list(exp_options.keys()))

    if selected_id:
        exp_meta = exp_options[selected_id]
        raw_payload = ExperimentService.load_raw_experiment_file(exp_meta["file_path"])

        c_det1, c_det2 = st.columns([1, 2])
        with c_det1:
            st.markdown(f"**Experiment ID:** `{exp_meta['experiment_id']}`")
            st.markdown(f"**File Location:** `{exp_meta['file_path']}`")
            st.markdown(f"**Last Modified:** `{exp_meta['timestamp']}`")
            st.download_button(
                label="⬇️ Download Raw JSON",
                data=json.dumps(raw_payload, indent=2),
                file_name=f"{selected_id}.json",
                mime="application/json",
            )

        with c_det2:
            st.json(raw_payload)

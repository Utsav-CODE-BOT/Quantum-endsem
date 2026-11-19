"""Page 8: Research Documentation & System Knowledge Base."""

from __future__ import annotations

import datetime
from pathlib import Path
import platform
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Documentation | Quantum Architecture Optimizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme stylesheet
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

# Header
st.markdown(
    """
    <div style="margin-bottom: 1.25rem;">
        <h1 style="color: #f8fafc; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem;">
            📚 Research Documentation & System Specifications
        </h1>
        <div style="font-size: 1.05rem; color: #94a3b8; font-weight: 500;">
            Interactive technical specifications, theoretical formulations, empirical findings, and operational guides
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Locate workspace root
workspace_root = Path(__file__).resolve().parent.parent.parent

# Document catalog
DOCUMENTS = {
    "System Architecture": {
        "path": workspace_root / "docs" / "architecture.md",
        "description": "Component boundaries, data flow pipelines, quantum circuit factory, and evolutionary search contracts.",
        "icon": "🏗️",
    },
    "Theoretical Methodology": {
        "path": workspace_root / "docs" / "methodology.md",
        "description": "Mathematical formulations for quantum encoding, ansatz definitions, COBYLA optimization, and multi-objective fitness.",
        "icon": "📐",
    },
    "Experimental Protocol": {
        "path": workspace_root / "docs" / "experiment_protocol.md",
        "description": "Zero-data-leakage PCA procedures, stratified splitting, evaluation metrics, and reproducibility guarantees.",
        "icon": "🧪",
    },
    "NISQ Limitations & Considerations": {
        "path": workspace_root / "docs" / "limitations.md",
        "description": "Barren plateau considerations, two-qubit gate noise scaling, simulator constraints, and scalability analysis.",
        "icon": "⚠️",
    },
    "Empirical Research Report": {
        "path": workspace_root / "experiments" / "results" / "RESEARCH_REPORT.md",
        "description": "Final experimental findings, baseline comparisons, noise resilience profiles, and ablation study conclusions.",
        "icon": "📊",
    },
    "Phase 13 Backend Audit": {
        "path": workspace_root / "docs" / "PHASE_13_BACKEND_AUDIT.md",
        "description": "Audit of all existing Python modules, model interfaces, data artifacts, and runtime benchmarks.",
        "icon": "🔍",
    },
    "Phase 13 Implementation Plan": {
        "path": workspace_root / "docs" / "PHASE_13_IMPLEMENTATION_PLAN.md",
        "description": "Architectural design, service adapter layers, and implementation blueprint for the Research Dashboard.",
        "icon": "🗺️",
    },
    "Framework User Guide": {
        "path": workspace_root / "docs" / "user_guide.md",
        "description": "CLI command-line usage, pipeline execution steps, and hyperparameter customization options.",
        "icon": "📖",
    },
    "Installation & Environment": {
        "path": workspace_root / "docs" / "installation.md",
        "description": "Python virtual environment setup, pinned dependency installation, and verification commands.",
        "icon": "⚙️",
    },
}

# Layout: Document Selector & Content
col_select, col_meta = st.columns([2, 1])

with col_select:
    selected_doc_name = st.selectbox(
        "Select Document to Read:",
        options=list(DOCUMENTS.keys()),
        index=0,
        format_func=lambda x: f"{DOCUMENTS[x]['icon']}  {x}",
    )

selected_info = DOCUMENTS[selected_doc_name]
doc_path: Path = selected_info["path"]

with col_meta:
    if doc_path.exists():
        stat = doc_path.stat()
        file_size_kb = stat.st_size / 1024
        modified_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        st.markdown(
            f"""
            <div style="background-color: #131c31; border: 1px solid #1e293b; border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; color: #94a3b8; margin-top: 1.5rem;">
                <span style="color: #06b6d4; font-weight: 600;">Size:</span> {file_size_kb:.1f} KB &nbsp;|&nbsp; 
                <span style="color: #06b6d4; font-weight: 600;">Updated:</span> {modified_time}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"File not found: `{doc_path.name}`")

st.markdown("---")

# Document Viewer with Search
if doc_path.exists():
    with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Search / Filter
    search_col, dl_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input(
            "Filter or Search within Document:",
            placeholder="Type a keyword (e.g. 'Pareto', 'COBYLA', 'fidelity', 'CX gates')...",
        )
    with dl_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Markdown",
            data=content,
            file_name=doc_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

    if search_query.strip():
        q = search_query.strip().lower()
        lines = content.split("\n")
        matching_lines = [
            (idx + 1, line)
            for idx, line in enumerate(lines)
            if q in line.lower()
        ]
        
        st.info(f"Found **{len(matching_lines)}** lines matching '{search_query}'.")
        if matching_lines:
            with st.expander("Search Results & Excerpts", expanded=True):
                for lineno, line in matching_lines[:15]:
                    highlighted = line.replace(
                        search_query, f"<mark style='background:#f59e0b; color:#0f172a; padding:0 3px; border-radius:2px;'>{search_query}</mark>"
                    )
                    st.markdown(
                        f"<div style='font-family: monospace; font-size: 0.82rem; margin-bottom: 0.3rem;'><span style='color: #06b6d4;'>L{lineno}:</span> {highlighted}</div>",
                        unsafe_allow_html=True,
                    )
                if len(matching_lines) > 15:
                    st.caption(f"... and {len(matching_lines) - 15} more matches.")

    # Render Main Content
    st.markdown(
        f"""
        <div style="background-color: #0d1527; border: 1px solid #1e293b; border-radius: 10px; padding: 1.5rem 2rem; margin-top: 1rem;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error(f"Target documentation file does not exist at `{doc_path}`.")

st.markdown("---")

# Quick Reference Section
col_ref, col_env = st.columns(2)

with col_ref:
    st.subheader("⚡ Quick CLI Commands Reference")
    st.markdown(
        """
| Pipeline Task | Command | Description |
|---|---|---|
| **Data Prep** | `python -m src.data.dataset` | Downloads Fashion-MNIST, fits PCA, generates splits |
| **Baseline VQC** | `python -m src.models.vqc` | Trains standard baseline VQC with COBYLA |
| **Arch Search** | `python -m src.search.evolutionary` | Runs evolutionary multi-objective architecture search |
| **Noise Eval** | `python -m src.noise.noise_models` | Benchmarks models across NISQ noise models |
| **Benchmarking** | `python -m src.evaluation.benchmark` | Evaluates all 7 classical and quantum models |
| **Full Pipeline**| `python run_experiments.py` | Orchestrates complete research pipeline end-to-end |
| **Launch UI** | `streamlit run ui/app.py` | Launches this interactive Research Dashboard |
| **Run Tests** | `pytest tests/ -v` | Executes full automated test suite |
        """
    )

with col_env:
    st.subheader("🖥️ Environment & System Specifications")
    st.markdown(
        f"""
- **Python Version:** `{platform.python_version()}` (`{platform.python_implementation()}`)
- **Operating System:** `{platform.system()} {platform.release()} ({platform.machine()})`
- **Qiskit Core:** `1.4.2`
- **Qiskit Aer:** `0.16.4`
- **Qiskit Machine Learning:** `0.8.4`
- **Streamlit Version:** `{st.__version__}`
- **Working Directory:** `{workspace_root.name}`
- **Dataset:** Fashion-MNIST (Binary: T-shirt/top vs. Ankle boot)
- **Primary Optimization:** COBYLA (SciPy / Qiskit)
        """
    )

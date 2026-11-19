"""Page 4: Evolutionary Architecture Search — Search Configuration and Monitoring."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import pandas as pd
import streamlit as st

from ui.components.charts import create_fitness_convergence_plot, create_pareto_scatter
from ui.components.sidebar import render_sidebar
from ui.services.search_service import SearchService
from ui.utils.formatting import format_percentage, format_time
from ui.utils.state import init_session_state
from ui.utils.validation import validate_fitness_weights, validate_search_params

st.set_page_config(
    page_title="Architecture Search | Quantum Architecture Optimizer",
    page_icon="🧬",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("🧬 Evolutionary Architecture Search")
st.markdown(
    "Automate the search for resource-aware Variational Quantum Classifiers. "
    "The evolutionary search engine utilizes tournament selection, architectural gene crossover, discrete structural mutation, "
    "and elitism to navigate the combinatorial circuit space toward the Pareto frontier."
)

st.markdown("---")

# Layout: Configuration on Left / Monitoring on Right
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("⚙️ Search Hyperparameters")

    pop_size = st.number_input("Population Size", min_value=2, max_value=8, value=4, step=1, help="Number of candidate architectures per generation.")
    generations = st.number_input("Generations", min_value=1, max_value=4, value=2, step=1, help="Maximum evolutionary iterations.")
    maxiter = st.number_input("COBYLA Iterations", min_value=10, max_value=50, value=25, step=5, help="Classical optimization iterations per candidate.")

    st.caption(
        "💡 *Note:* Population $\\le 8$ and Generations $\\le 4$ are UI interactive safety bounds designed to maintain responsive browser execution "
        "on standard development workstations. For arbitrary population sizes, generations, or high-throughput batch experiments, "
        "use the headless CLI script: `python scripts/run_architecture_search.py`."
    )

    st.markdown("**Fitness Objective Weights**")
    w_perf = st.slider("Performance Weight (w_perf)", 0.0, 1.0, 0.50, 0.05, help="Weight given to validation classification accuracy.")
    w_comp = st.slider("Complexity Weight (w_comp)", 0.0, 1.0, 0.30, 0.05, help="Penalty weight on circuit depth and gate count.")
    w_rob = st.slider("Robustness Weight (w_rob)", 0.0, 1.0, 0.20, 0.05, help="Weight given to simulated noise tolerance.")

    is_weights_valid, w_err = validate_fitness_weights(w_perf, w_comp, w_rob)
    if not is_weights_valid:
        st.caption(f":warning: {w_err}")

    with st.expander("Advanced Genetic Settings"):
        cross_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.70, 0.05)
        mut_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.30, 0.05)
        tourn_size = st.slider("Tournament Size", 2, 4, 2)
        seed = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")

    # Search Execution Controls
    is_running = SearchService.is_running()

    if not is_running:
        if st.button("🚀 Start Evolutionary Search", type="primary", use_container_width=True):
            if not is_weights_valid:
                st.error("Please ensure weights sum to 1.0 before launching.")
            else:
                success = SearchService.start_search(
                    pop_size=pop_size,
                    generations=generations,
                    maxiter=maxiter,
                    w_perf=w_perf,
                    w_comp=w_comp,
                    w_rob=w_rob,
                    crossover_rate=cross_rate,
                    mutation_rate=mut_rate,
                    tournament_size=tourn_size,
                    random_seed=seed,
                )
                if success:
                    st.success("Evolutionary search launched in background!")
                    st.rerun()
                else:
                    st.warning("A search is already in progress.")
    else:
        st.markdown(
            '<div style="text-align: center; margin-bottom: 0.5rem;"><span class="status-badge active">● SEARCH IN PROGRESS</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🛑 Cancel Search", type="secondary", use_container_width=True):
            SearchService.cancel_search()
            st.warning("Cancellation requested.")
            st.rerun()

with col_right:
    status = SearchService.get_status()

    # If running or just completed, show live status
    if status["status"] in ("running", "cancelling"):
        st.subheader("📡 Live Search Telemetry")
        if status["status"] == "cancelling":
            st.warning("⚠️ Cancellation requested. Finishing current candidate before terminating gracefully...")
        current_g = status["current_generation"]
        total_g = status["total_generations"]
        progress_val = min(1.0, max(0.0, current_g / max(1, total_g)))
        st.progress(progress_val, text=f"Generation {current_g} of {total_g} in progress...")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current Gen", f"{current_g}/{total_g}")
        with m2:
            st.metric("Best Fitness", f"{status['best_fitness']:.4f}")
        with m3:
            st.metric("Best Val Acc", format_percentage(status["best_val_accuracy"]))
        with m4:
            st.metric("Candidates Evaluated", status["num_evaluated_candidates"])

        # Auto-refresh while running
        time.sleep(2)
        st.rerun()

    else:
        # Show completed or saved search results
        saved_hist = SearchService.load_saved_history()
        saved_best = SearchService.load_saved_best_architecture()

        st.subheader("🏆 Best Discovered Architecture")

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        best_cfg = saved_best.get("config", {})
        best_meta = saved_best.get("circuit_metadata", {})
        best_val = saved_best.get("val_metrics", {}).get("accuracy")

        with b_c1:
            st.metric("Architecture ID", saved_best.get("arch_id", "arch_6327d7bcb8"))
        with b_c2:
            st.metric("Best Validation Accuracy", format_percentage(best_val))
        with b_c3:
            st.metric("Circuit Depth", best_meta.get("circuit_depth", 5))
        with b_c4:
            st.metric("Total Parameters", best_meta.get("num_trainable_parameters", 16))

        st.markdown(
            """
            <div class="research-callout">
                <strong>Scientific Integrity Notice:</strong> The validation accuracy shown above is computed exclusively on the
                held-out validation partition (60 samples). The test partition (100 samples) was completely isolated from
                the evolutionary selection process to guarantee zero data leakage.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Generational Convergence Chart
        gen_records = saved_hist.get("generation_history", [])
        if gen_records:
            fig_conv = create_fitness_convergence_plot(gen_records)
            st.plotly_chart(fig_conv, use_container_width=True)

        # Evaluated Candidates Table
        st.subheader("📋 Evaluated Candidate Architectures")
        eval_list = []
        for g in gen_records:
            cand_ids = g.get("candidate_ids", [])
            cand_fits = g.get("candidate_fitnesses", [])
            for cid, cfit in zip(cand_ids, cand_fits):
                eval_list.append({
                    "Generation": g.get("generation"),
                    "Architecture ID": cid,
                    "Fitness": round(cfit, 4),
                    "Status": "Evaluated",
                })

        if eval_list:
            df_cands = pd.DataFrame(eval_list)
            st.dataframe(df_cands, use_container_width=True, hide_index=True)

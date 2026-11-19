"""Page 5: Quantum Noise Analysis — Robustness under Simulated Quantum Errors."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.evaluation.noise_evaluator import NoiseModelConfig
from ui.components.charts import create_noise_degradation_plot
from ui.components.sidebar import render_sidebar
from ui.services.circuit_service import CircuitService
from ui.services.noise_service import NoiseService
from ui.utils.formatting import format_percentage
from ui.utils.state import init_session_state

st.set_page_config(
    page_title="Noise Analysis | Quantum Architecture Optimizer",
    page_icon="📡",
    layout="wide",
)

# Apply global styling
css_path = Path(__file__).resolve().parent.parent / "styles" / "theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

init_session_state()
render_sidebar()

st.title("📡 Quantum Hardware Noise Analysis")
st.markdown(
    "Evaluate candidate VQC architectures under simulated quantum hardware errors using "
    "**Qiskit Aer's density matrix backend** with depolarizing gate errors and measurement readout bit-flips."
)

st.markdown("---")

# Standard Noise Tiers Description
st.subheader("⚙️ Standard NISQ Noise Regimes")
tiers = NoiseService.get_canonical_tiers()

tier_cols = st.columns(3)
for col, (t_name, t_cfg) in zip(tier_cols, tiers.items()):
    with col:
        st.markdown(
            f"""
            <div style="background: #131c31; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem;">
                <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;">{t_name}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-family: monospace; line-height: 1.6;">
                    1-Qubit Error (p<sub>1q</sub>): <strong>{t_cfg.single_qubit_error*100:.2f}%</strong><br>
                    2-Qubit Error (p<sub>2q</sub>): <strong>{t_cfg.two_qubit_error*100:.2f}%</strong><br>
                    Readout Error (p<sub>meas</sub>): <strong>{t_cfg.measurement_error*100:.2f}%</strong><br>
                    Shots: <strong>{t_cfg.shots:,}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Section 1: Saved Benchmark Noise Trajectory
st.subheader("📈 Noise Degradation & Accuracy Retention")

saved_noise = NoiseService.load_saved_results()
if saved_noise:
    fig_noise = create_noise_degradation_plot(saved_noise)
    st.plotly_chart(fig_noise, use_container_width=True)

    # Summary Comparison Table
    st.markdown("### 📋 Empirical Noise Benchmark Table")

    rows = []
    for model_key, label in [("baseline", "Baseline VQC (Depth 12, 6 CX)"), ("optimized", "Search-Optimized (Depth 5, 0 CX)")]:
        for r in saved_noise.get("benchmarks", {}).get(model_key, []):
            cfg = r["noise_config"]
            ideal_acc = r["ideal_metrics"]["accuracy"]
            noisy_acc = r["noisy_metrics"]["accuracy"]
            drop = r["accuracy_drop"]
            ret = r["relative_accuracy_retention"]
            sens = r.get("noise_sensitivity_index", 0.0)

            rows.append({
                "Architecture": label,
                "Noise Regime": cfg["name"],
                "1Q Error": f"{cfg['single_qubit_error']*100:.2f}%",
                "2Q Error": f"{cfg['two_qubit_error']*100:.2f}%",
                "Readout Error": f"{cfg['measurement_error']*100:.2f}%",
                "Ideal Acc": format_percentage(ideal_acc),
                "Noisy Acc": format_percentage(noisy_acc),
                "Acc Drop": f"{drop*100:+.2f}%",
                "Retention Rate": format_percentage(ret),
                "Sensitivity Index": f"{sens:.4f}",
            })

    if rows:
        df_noise = pd.DataFrame(rows)
        st.dataframe(df_noise, use_container_width=True, hide_index=True)

st.markdown("---")

# Section 2: Interactive Real-Time Noise Evaluator
st.subheader("🧪 Run Live Noise Experiment on Custom Architecture")

col_eval1, col_eval2 = st.columns([1, 2])

with col_eval1:
    arch_options = CircuitService.get_canonical_architectures()
    chosen_name = st.selectbox("Select Target Architecture", list(arch_options.keys()))
    chosen_cfg = arch_options[chosen_name]

    chosen_tier_name = st.selectbox("Select Noise Regime", list(tiers.keys()))
    chosen_tier_cfg = tiers[chosen_tier_name]

    eval_maxiter = st.slider("Training Iterations (COBYLA)", 10, 35, 20, 5)

    run_btn = st.button("⚡ Execute Live Noise Evaluation", type="primary", use_container_width=True)

with col_eval2:
    if run_btn:
        with st.spinner(f"Training VQC and simulating on Aer density matrix ({chosen_tier_name})..."):
            try:
                report = NoiseService.evaluate_architecture_noise(
                    arch_config=chosen_cfg,
                    noise_config=chosen_tier_cfg,
                    maxiter=eval_maxiter,
                )

                st.success("Noise Evaluation Completed Successfully!")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Ideal Accuracy", format_percentage(report.ideal_metrics.accuracy))
                with m2:
                    st.metric("Noisy Accuracy", format_percentage(report.noisy_metrics.accuracy))
                with m3:
                    st.metric("Accuracy Drop", f"{report.accuracy_drop*100:+.2f}%")
                with m4:
                    st.metric("Retention Rate", format_percentage(report.relative_accuracy_retention))

                with st.expander("Full Evaluation Payload"):
                    st.json(report.to_dict())

            except Exception as e:
                st.error(f"Noise evaluation error: {e}")
    else:
        st.info("Click **Execute Live Noise Evaluation** to train this candidate and test under simulated noise channels.")

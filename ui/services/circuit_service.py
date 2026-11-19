"""Quantum circuit compilation, metadata extraction, and rendering service."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from src.optimization.architecture_generator import ArchitectureGenerator
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata, VQCCircuitGenerator
from src.utils.config_loader import get_project_root


class CircuitService:
    """Service providing quantum circuit compilation and multi-format rendering."""

    @staticmethod
    def compile_from_config(config: ArchitectureConfig) -> tuple[QuantumCircuit, CircuitMetadata]:
        """Compile a complete VQC circuit and extract quantitative resource metrics.

        Args:
            config: ArchitectureConfig specifying qubits, depth, rotations, and entanglement.

        Returns:
            Tuple of (QuantumCircuit, CircuitMetadata).
        """
        gen, meta = ArchitectureGenerator.compile_circuit(config)
        return gen.full_circuit, meta

    @staticmethod
    def draw_text(circuit: QuantumCircuit) -> str:
        """Render circuit as clean Unicode text diagram."""
        return str(circuit.draw(output="text"))

    @staticmethod
    def draw_matplotlib(circuit: QuantumCircuit, scale: float = 0.85) -> plt.Figure:
        """Render circuit as Matplotlib publication-grade diagram."""
        try:
            fig = circuit.draw(
                output="mpl",
                scale=scale,
                style={
                    "backgroundcolor": "#0d1322",
                    "linecolor": "#94a3b8",
                    "textcolor": "#f8fafc",
                    "gatetextcolor": "#0f172a",
                    "gatefacecolor": "#818cf8",
                    "barrierfacecolor": "#334155",
                },
            )
            return fig
        except Exception:
            # Clean fallback if optional latex/rendering libraries are missing
            text_rep = str(circuit.draw(output="text"))
            lines = text_rep.split("\n")
            max_len = max(len(l) for l in lines) if lines else 40
            fig, ax = plt.subplots(figsize=(max(8, max_len * 0.12), max(3.5, len(lines) * 0.35)))
            fig.patch.set_facecolor("#0d1322")
            ax.set_facecolor("#0d1322")
            ax.text(
                0.02,
                0.95,
                text_rep,
                family="monospace",
                fontsize=9,
                color="#38bdf8",
                verticalalignment="top",
                transform=ax.transAxes,
            )
            ax.axis("off")
            return fig

    @staticmethod
    def export_circuit_png_bytes(circuit: QuantumCircuit) -> bytes:
        """Export circuit drawing directly to PNG bytes for user download."""
        fig = CircuitService.draw_matplotlib(circuit, scale=1.0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def get_canonical_architectures() -> dict[str, ArchitectureConfig]:
        """Return canonical predefined research architectures for quick exploration."""
        return {
            "Search-Optimized (arch_6327d7bcb8)": ArchitectureConfig(
                num_qubits=4,
                num_layers=1,
                rotation_gates=("rx", "ry"),
                entanglement_type="circular",
                entanglement_freq=2,
            ),
            "Baseline Standard VQC": ArchitectureConfig(
                num_qubits=4,
                num_layers=2,
                rotation_gates=("ry", "rz"),
                entanglement_type="linear",
                entanglement_freq=1,
            ),
            "Fixed Shallow VQC": ArchitectureConfig(
                num_qubits=4,
                num_layers=1,
                rotation_gates=("ry",),
                entanglement_type="linear",
                entanglement_freq=1,
            ),
            "Fixed Deep VQC": ArchitectureConfig(
                num_qubits=4,
                num_layers=4,
                rotation_gates=("ry", "rz"),
                entanglement_type="circular",
                entanglement_freq=1,
            ),
        }

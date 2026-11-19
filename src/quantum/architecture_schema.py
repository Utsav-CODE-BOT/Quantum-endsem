"""Architecture schema representing candidate VQC circuit designs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ArchitectureConfig:
    """Immutable representation of a candidate VQC architecture gene."""

    num_qubits: int
    num_layers: int
    rotation_gates: tuple[str, ...]
    entanglement_type: str
    entanglement_freq: int = 1
    feature_rotation: str = "ry"
    measurement_strategy: str = "parity"

    @property
    def arch_id(self) -> str:
        """Deterministic unique architecture identifier derived from architectural attributes."""
        canonical_dict = {
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "rotation_gates": sorted(list(self.rotation_gates)),
            "entanglement_type": self.entanglement_type.lower(),
            "entanglement_freq": self.entanglement_freq,
            "feature_rotation": self.feature_rotation.lower(),
            "measurement_strategy": self.measurement_strategy.lower(),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return f"arch_{hashlib.sha256(encoded).hexdigest()[:10]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert architecture specification to dictionary."""
        d = asdict(self)
        d["arch_id"] = self.arch_id
        d["rotation_gates"] = list(self.rotation_gates)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureConfig":
        """Reconstruct architecture from dictionary."""
        return cls(
            num_qubits=int(data["num_qubits"]),
            num_layers=int(data["num_layers"]),
            rotation_gates=tuple(data["rotation_gates"]),
            entanglement_type=str(data["entanglement_type"]),
            entanglement_freq=int(data.get("entanglement_freq", 1)),
            feature_rotation=str(data.get("feature_rotation", "ry")),
            measurement_strategy=str(data.get("measurement_strategy", "parity")),
        )

"""Configuration loader for the Quantum Architecture Optimizer.

Loads YAML configuration files and provides a unified interface
for accessing experiment parameters across all modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory.

    Walks up from this file's location to find the directory
    containing 'configs/'.
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "configs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find project root (directory containing 'configs/'). "
        f"Searched from {Path(__file__).resolve().parent}"
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dictionary.

    Args:
        path: Absolute or project-relative path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


class ConfigManager:
    """Centralized configuration manager.

    Loads all project configuration files and provides dot-notation
    and dictionary access to parameters.
    """

    # Standard configuration file names
    CONFIG_FILES = [
        "dataset.yaml",
        "preprocessing.yaml",
        "quantum.yaml",
        "training.yaml",
        "evolutionary_search.yaml",
        "noise.yaml",
        "experiments.yaml",
    ]

    def __init__(self, config_dir: str | Path | None = None) -> None:
        """Initialize the configuration manager.

        Args:
            config_dir: Path to the configs directory.
                        Defaults to <project_root>/configs/.
        """
        if config_dir is None:
            config_dir = get_project_root() / "configs"
        self.config_dir = Path(config_dir)
        self._configs: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all standard configuration files."""
        for filename in self.CONFIG_FILES:
            filepath = self.config_dir / filename
            config_name = filename.replace(".yaml", "").replace(".yml", "")
            if filepath.exists():
                self._configs[config_name] = load_yaml(filepath)
            # Missing files are allowed — they just won't be in the config

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            section: Top-level config file name (without .yaml).
            key: Dot-separated key path (e.g., 'pca.n_components').
            default: Default value if the key is not found.

        Returns:
            The configuration value.

        Examples:
            >>> cfg.get('dataset', 'dataset.selected_classes')
            [0, 9]
            >>> cfg.get('preprocessing', 'preprocessing.pca.n_components')
            4
        """
        if section not in self._configs:
            return default

        data = self._configs[section]
        if key is None:
            return data

        keys = key.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_section(self, section: str) -> dict[str, Any]:
        """Get an entire configuration section.

        Args:
            section: Config file name (without .yaml).

        Returns:
            The section's configuration dictionary, or empty dict.
        """
        return self._configs.get(section, {})

    def override(self, section: str, key: str, value: Any) -> None:
        """Override a configuration value at runtime.

        Useful for experiment-specific parameter changes.

        Args:
            section: Config file name (without .yaml).
            key: Dot-separated key path.
            value: New value.
        """
        if section not in self._configs:
            self._configs[section] = {}

        keys = key.split(".")
        current = self._configs[section]
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return all configurations as a nested dictionary."""
        import copy
        return copy.deepcopy(self._configs)

    def save_snapshot(self, path: str | Path) -> None:
        """Save current configuration state to a YAML file.

        Used for experiment reproducibility.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        sections = list(self._configs.keys())
        return f"ConfigManager(sections={sections})"

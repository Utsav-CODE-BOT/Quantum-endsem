#!/usr/bin/env python
"""Download and inspect the Fashion-MNIST dataset.

Usage:
    python scripts/download_dataset.py [--classes 0 9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import FashionMNISTLoader
from src.utils.config_loader import ConfigManager
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Fashion-MNIST dataset")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Class labels to extract (e.g. 0 9)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO", log_dir="experiments/logs", log_file="download_dataset.log")
    cfg = ConfigManager()

    if args.classes is not None:
        selected_classes = args.classes
    else:
        selected_classes = cfg.get("dataset", "dataset.selected_classes", [0, 9])

    logger.info("Initializing loader for classes: %s", selected_classes)
    loader = FashionMNISTLoader(selected_classes=selected_classes)

    train_data = loader.load(split="train")
    test_data = loader.load(split="test")

    print("\n" + "=" * 60)
    print("FASHION-MNIST DATASET DOWNLOAD & INSPECTION REPORT")
    print("=" * 60)
    print(f"Selected Classes:   {selected_classes}")
    for orig_cls, new_cls in train_data.label_mapping.items():
        name = train_data.class_names[new_cls]
        print(f"  Class {orig_cls} ('{name}') -> Remapped to {new_cls}")
    print(f"Training Samples:   {len(train_data.images)} (images: {train_data.images.shape})")
    print(f"Test Samples:       {len(test_data.images)} (images: {test_data.images.shape})")
    print(f"Image Resolution:   {train_data.images.shape[1]}x{train_data.images.shape[2]}")
    print(f"Pixel Range:        [{train_data.images.min()}, {train_data.images.max()}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

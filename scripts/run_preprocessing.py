#!/usr/bin/env python
"""Execute the complete data acquisition, splitting, and preprocessing pipeline.

Usage:
    python scripts/run_preprocessing.py [--n-components 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import FashionMNISTLoader
from src.data.dataset_splitter import DatasetSplitter
from src.data.data_validator import DataValidator
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.utils.config_loader import ConfigManager, get_project_root
from src.utils.logging_config import get_logger, setup_logging
from src.utils.seed_manager import SeedManager

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--n-components", type=int, default=None, help="PCA components (e.g. 4, 6, 8)")
    args = parser.parse_args()

    setup_logging(level="INFO", log_dir="experiments/logs", log_file="run_preprocessing.log")
    cfg = ConfigManager()
    seed_mgr = SeedManager(global_seed=cfg.get("experiments", "experiments.global_seed", 42))

    # 1. Load dataset
    loader = FashionMNISTLoader.from_config(cfg)
    train_raw = loader.load(split="train")
    test_raw = loader.load(split="test")

    # 2. Split dataset
    splitter = DatasetSplitter(
        samples_per_class_train=cfg.get("dataset", "dataset.samples_per_class_train", 100),
        samples_per_class_val=cfg.get("dataset", "dataset.samples_per_class_val", 30),
        samples_per_class_test=cfg.get("dataset", "dataset.samples_per_class_test", 50),
        seed_manager=seed_mgr,
    )
    splits = splitter.create_splits(train_raw, test_raw)
    DataValidator.validate_splits(splits)

    # 3. Preprocess
    n_components = args.n_components or cfg.get("preprocessing", "preprocessing.pca.n_components", 4)
    pipeline = PreprocessingPipeline(
        n_components=n_components,
        random_state=seed_mgr.derive_seed("pca_pipeline"),
    )
    preprocessed = pipeline.transform_splits(splits)

    # 4. Save processed arrays and metadata
    proj_root = get_project_root()
    processed_dir = proj_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = proj_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        processed_dir / "fashion_mnist_quantum_ready.npz",
        X_train=preprocessed.X_train,
        y_train=preprocessed.y_train,
        X_val=preprocessed.X_val,
        y_val=preprocessed.y_val,
        X_test=preprocessed.X_test,
        y_test=preprocessed.y_test,
    )

    # Save pipeline object and metadata json
    pipeline.save(metadata_dir / "preprocessing_pipeline.pkl")
    meta = pipeline.get_metadata()
    meta["split_summary"] = splits.summary()
    meta["class_mapping"] = splits.label_mapping
    meta["class_names"] = splits.class_names
    with open(metadata_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"PCA Components:          {preprocessed.n_components}")
    print(f"Explained Variance Ratio:{np.round(meta['pca_metadata']['explained_variance_ratio'], 4)}")
    print(f"Cumulative Variance:     {meta['pca_metadata']['cumulative_variance']:.4f}")
    print(f"Training Data Shape:     {preprocessed.X_train.shape} (Labels: {preprocessed.y_train.shape})")
    print(f"Validation Data Shape:   {preprocessed.X_val.shape} (Labels: {preprocessed.y_val.shape})")
    print(f"Test Data Shape:         {preprocessed.X_test.shape} (Labels: {preprocessed.y_test.shape})")
    print(f"Rotation Angle Range:    [{preprocessed.X_train.min():.4f}, {preprocessed.X_train.max():.4f}] rad")
    print(f"Saved Processed Data:    {processed_dir / 'fashion_mnist_quantum_ready.npz'}")
    print(f"Saved Metadata:          {metadata_dir / 'preprocessing_metadata.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

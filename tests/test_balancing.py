#!/usr/bin/env python3
"""Test script for balancing comparison functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.skinmap.evaluation.downstream import (
    evaluate_downstream_with_balancing_comparison,
)

# Test imports
from src.skinmap.utils.balancing import balance_dataset
from src.skinmap.utils.metadata import evaluate_metadata_with_balancing_comparison


def test_balance_dataset():
    """Test the balance_dataset function."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)  # Highly imbalanced

    # Test undersample
    X_under, y_under = balance_dataset(X, y, strategy="undersample", random_state=42)
    assert len(y_under) <= len(y), "Undersampling should reduce samples"

    # Test oversample
    X_over, y_over = balance_dataset(X, y, strategy="oversample", random_state=42)
    assert len(y_over) >= len(y), "Oversampling should increase samples"

    # Test none
    X_none, y_none = balance_dataset(X, y, strategy="none", random_state=42)
    assert len(y_none) == len(y), "Strategy 'none' should not change sample count"


def test_metadata_balancing_comparison():
    """Test metadata balancing comparison function."""
    np.random.seed(42)
    n_samples = 200
    embeddings = np.random.randn(n_samples, 128)

    # Create imbalanced classification task
    gender_values = ["male"] * 140 + ["female"] * 40 + ["other"] * 20
    np.random.shuffle(gender_values)

    df = pd.DataFrame(
        {
            "gender": gender_values,
            "age": np.random.randint(18, 80, n_samples),
            "body_region": np.random.choice(["head", "torso", "limbs"], n_samples),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results_df = evaluate_metadata_with_balancing_comparison(
            embeddings=embeddings,
            df=df,
            attributes=["gender", "age", "body_region"],
            test_size=0.2,
            random_state=42,
            output_dir=tmpdir,
        )

        assert not results_df.empty, "Results should not be empty"
        assert set(results_df["strategy"].unique()) == {
            "none",
            "undersample",
            "oversample",
        }
        assert (
            "age" not in results_df["attribute"].values
        )  # Regression should be skipped
        assert Path(tmpdir, "analysis", "metadata_balancing_comparison.csv").exists()


def test_downstream_balancing_comparison():
    """Test downstream balancing comparison function."""
    np.random.seed(42)
    n_samples = 200
    embeddings = np.random.randn(n_samples, 128)

    class MockDataset:
        def __init__(self):
            self.meta_data = pd.DataFrame(
                {
                    "malignant": np.random.choice(
                        ["yes", "no"], n_samples, p=[0.2, 0.8]
                    ),
                    "skin_tone": np.random.choice(
                        ["light", "medium", "dark"], n_samples, p=[0.6, 0.3, 0.1]
                    ),
                }
            )

    dataset = MockDataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_df = evaluate_downstream_with_balancing_comparison(
            embeddings=embeddings,
            dataset=dataset,
            classification_cols=["malignant", "skin_tone"],
            test_size=0.2,
            random_state=42,
            output_dir=tmpdir,
            dataset_name="MockDataset",
        )

        assert not results_df.empty, "Results should not be empty"
        assert set(results_df["strategy"].unique()) == {
            "none",
            "undersample",
            "oversample",
        }
        assert set(results_df["task"].unique()) == {"malignant", "skin_tone"}
        assert Path(tmpdir, "analysis", "downstream_balancing_MockDataset.csv").exists()

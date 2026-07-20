"""Dataset balancing utilities for handling class imbalance.

Provides under-sampling and over-sampling strategies for creating balanced datasets.
"""

from typing import Literal, Tuple

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger


def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    strategy: Literal["undersample", "oversample", "none"] = "none",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance dataset using under-sampling or over-sampling.

    Args:
        X: Feature matrix (N, D)
        y: Labels (N,)
        strategy: Balancing strategy ("undersample", "oversample", or "none")
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    if strategy == "none":
        return X, y

    # Count original class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Original class distribution: {dict(zip(unique, counts))}")

    if strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        logger.info(f"Applied under-sampling: {len(y)} -> {len(y_balanced)} samples")
    elif strategy == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        logger.info(f"Applied over-sampling: {len(y)} -> {len(y_balanced)} samples")
    else:
        raise ValueError(
            f"Unknown balancing strategy: {strategy}. "
            f"Choose from 'undersample', 'oversample', or 'none'."
        )

    # Count balanced class distribution
    unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)
    logger.info(
        f"Balanced class distribution: {dict(zip(unique_balanced, counts_balanced))}"
    )

    return X_balanced, y_balanced

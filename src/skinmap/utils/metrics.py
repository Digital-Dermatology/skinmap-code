"""Learned metric transformations for embedding spaces."""

import time
import traceback
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier


def compute_learned_metric_transformation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    max_samples: int = 250_000,
) -> Optional[np.ndarray]:
    """Learn a metric transformation from SGDClassifier weights.

    Trains a classifier on the provided embeddings and labels, then computes
    transformation L such that M = W^T @ W = L^T @ L, where W are the
    classifier coefficients. This learned metric can then be used for NN search.

    Args:
        embeddings: Embedding vectors (N, d)
        labels: Class labels (N,)
        random_state: Random seed for reproducibility
        max_samples: Maximum samples to use for training to avoid OOM

    Returns:
        L: Transformation matrix of shape (k, d) where k <= d
        None if computation fails
    """
    try:
        N, d = embeddings.shape
        logger.info(f"Learning metric transformation from {N} samples with dim={d}")

        # Subsample if needed to avoid OOM
        if N > max_samples:
            logger.info(
                f"Subsampling {max_samples} from {N} samples for metric learning"
            )
            np.random.seed(random_state)
            indices = np.random.choice(N, max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            N = max_samples

        # Use SGDClassifier - faster than LogisticRegression for high-dim data
        start = time.time()

        try:
            clf = SGDClassifier(
                loss="log_loss",  # Logistic regression loss
                penalty="l2",
                alpha=0.0001,
                max_iter=1_000,
                tol=1e-3,
                random_state=random_state,
                n_jobs=-1,
                learning_rate="optimal",
            )
            clf.fit(embeddings, labels)
        except (OSError, RuntimeError) as exc:
            logger.warning(
                f"Parallel SGD failed ({exc}); retrying with single-threaded fit"
            )
            clf = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.0001,
                max_iter=1_000,
                tol=1e-3,
                random_state=random_state,
                n_jobs=1,
                learning_rate="optimal",
            )
            clf.fit(embeddings, labels)

        elapsed = time.time() - start
        W = clf.coef_  # (K, d) - K classes, d dimensions
        logger.info(
            f"Trained SGDClassifier in {elapsed:.2f}s: "
            f"{W.shape[0]} classes, {W.shape[1]} dimensions"
        )

        # Handle both binary and multi-class cases
        if W.shape[0] == 1:
            W = np.vstack([-W, W])

        # SVD: W = U @ diag(s) @ Vt
        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        # Build L such that W^T @ W = L^T @ L (no extra square root)
        L = np.diag(s) @ Vt

        logger.info(f"Computed metric transformation: L shape {L.shape}")
        logger.info(
            f"  Singular values: min={s.min():.4f}, "
            f"max={s.max():.4f}, mean={s.mean():.4f}"
        )

        return L

    except Exception as e:
        logger.error(f"Failed to compute metric transformation: {e}")
        logger.error(traceback.format_exc())
        return None

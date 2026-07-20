"""Metadata prediction utilities using XGBoost on embeddings."""

import os
import pickle
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .balancing import balance_dataset
from .bootstrap import (
    compute_classification_metrics_with_bootstrap,
    compute_regression_metrics_with_bootstrap,
    format_metric_with_ci,
)


def _create_predictor_model(
    is_multilabel: bool, is_regression: bool, random_state: int
):
    """Create predictor model with consistent configuration.

    This ensures evaluation and final models use identical settings.

    Args:
        is_multilabel: Whether task is multi-label classification
        is_regression: Whether task is regression
        random_state: Random seed for reproducibility

    Returns:
        Configured model instance
    """
    if is_multilabel:
        base_classifier = LogisticRegression(
            max_iter=5_000,
            random_state=random_state,
        )
        return MultiOutputClassifier(base_classifier)
    elif is_regression:
        return Ridge(alpha=1.0, random_state=random_state)
    else:
        return LogisticRegression(
            max_iter=5_000,
            random_state=random_state,
        )


def predict_missing_metadata(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    attributes: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    balancing_strategy: Literal["none", "undersample", "oversample"] = "none",
    bootstrap_n_iterations: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predict missing metadata attributes using linear models on embeddings.

    Supports single-label classification, multi-label classification, and regression.
    Multi-label classification is detected automatically when values are lists.

    Uses LogisticRegression for classification and Ridge for regression.

    Args:
        embeddings: Image embeddings array (N, D)
        df: Metadata dataframe with N rows
        attributes: List of attribute column names to predict
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        output_dir: Directory to save trained models (optional)
        balancing_strategy: Strategy for balancing training data ("none", "undersample", "oversample")
                           Only applies to single-label classification tasks.
        bootstrap_n_iterations: Number of bootstrap iterations for uncertainty estimation (0 = disabled)

    Returns:
        Tuple of (updated dataframe with predictions, metrics dataframe, confusion matrices dict)
        - Updated df has new columns: {attribute}_pred with predictions
        - Metrics df contains evaluation metrics for each attribute (with CIs if bootstrap enabled)
        - Confusion matrices dict: {attribute: {"confusion_matrix": cm, "label_encoder": encoder}}
          (only for single-label classification tasks)
    """
    all_metrics = []
    all_confusion_matrices = {}
    df_output = df.copy()

    # Create model output directory if specified
    if output_dir:
        model_dir = os.path.join(output_dir, "prediction_models")
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = None

    for attribute in attributes:
        if attribute not in df.columns:
            logger.warning(f"Attribute {attribute} not found in dataframe, skipping")
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing attribute: {attribute}")

        # Identify samples with known and missing values
        has_value = df[attribute].notna()
        missing_value = df[attribute].isna()

        n_has_value = has_value.sum()
        n_missing = missing_value.sum()

        logger.info(f"Samples with {attribute}: {n_has_value}")
        logger.info(f"Samples missing {attribute}: {n_missing}")

        if n_has_value < 10:
            logger.warning(
                f"Too few samples with {attribute} ({n_has_value}), skipping"
            )
            continue

        if n_missing == 0:
            logger.info(f"No missing values for {attribute}, skipping")
            continue

        # Determine task type
        is_regression = attribute == "age"

        # Check if this is multi-label classification (values are lists)
        is_multilabel = False
        if not is_regression:
            sample_values = df.loc[has_value, attribute].head(10)
            is_multilabel = any(isinstance(val, list) for val in sample_values)

        if is_multilabel:
            logger.info("Task type: Multi-label Classification")
        elif is_regression:
            logger.info("Task type: Regression")
        else:
            logger.info("Task type: Single-label Classification")

        # Get embeddings and labels for samples with known values
        X_known = embeddings[has_value]
        y_known = df.loc[has_value, attribute].values

        # Encode labels based on task type
        mlb = None
        le = None

        if is_multilabel:
            mlb = MultiLabelBinarizer()
            y_known_encoded = mlb.fit_transform(y_known)
            n_labels = len(mlb.classes_)
            logger.info(f"Number of labels: {n_labels}")
            logger.info(f"Labels: {list(mlb.classes_)[:10]}...")
        elif not is_regression:
            le = LabelEncoder()
            y_known_encoded = le.fit_transform(y_known)
            n_classes = len(le.classes_)
            logger.info(f"Number of classes: {n_classes}")
        else:
            y_known_encoded = y_known.astype(float)

        # Split for evaluation
        if is_multilabel:
            X_train, X_test, y_train, y_test = train_test_split(
                X_known,
                y_known_encoded,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_known,
                y_known_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_known_encoded if not is_regression else None,
            )

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train evaluation model with standard configuration
        logger.info("Training evaluation model...")
        model_eval = _create_predictor_model(is_multilabel, is_regression, random_state)
        model_eval.fit(X_train, y_train)

        # Evaluate
        y_pred = model_eval.predict(X_test)

        # Clip regression predictions to reasonable bounds
        clip_min, clip_max = None, None
        if is_regression:
            y_min = y_known_encoded.min()
            y_max = y_known_encoded.max()
            buffer = 0.05 * (y_max - y_min)
            clip_min = max(0, y_min - buffer)
            clip_max = y_max + buffer
            y_pred = np.clip(y_pred, clip_min, clip_max)
            logger.info(
                f"Clipping predictions to range [{clip_min:.1f}, {clip_max:.1f}]"
            )

        # Calculate metrics based on task type
        if is_multilabel:
            if bootstrap_n_iterations > 0:
                logger.info(
                    f"Computing bootstrap confidence intervals ({bootstrap_n_iterations} iterations)..."
                )
                bootstrap_metrics = compute_classification_metrics_with_bootstrap(
                    y_test,
                    y_pred,
                    n_iterations=bootstrap_n_iterations,
                    random_state=random_state,
                    is_multilabel=True,
                )

                metrics = {
                    "attribute": attribute,
                    "task_type": "multi-label",
                }

                # Add mean values and confidence intervals
                for metric_name, (
                    mean_val,
                    lower_ci,
                    upper_ci,
                ) in bootstrap_metrics.items():
                    metrics[metric_name] = mean_val
                    metrics[f"{metric_name}_ci_lower"] = lower_ci
                    metrics[f"{metric_name}_ci_upper"] = upper_ci

                logger.info(
                    f"Hamming Loss: {format_metric_with_ci(*bootstrap_metrics['hamming_loss'])}"
                )
                logger.info(
                    f"Jaccard (samples): {format_metric_with_ci(*bootstrap_metrics['jaccard_samples'])}"
                )
                logger.info(
                    f"F1 (macro): {format_metric_with_ci(*bootstrap_metrics['f1_macro'])}"
                )
            else:
                hamming = hamming_loss(y_test, y_pred)
                jaccard_micro = jaccard_score(
                    y_test,
                    y_pred,
                    average="micro",
                    zero_division=0,
                )
                jaccard_macro = jaccard_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
                jaccard_samples = jaccard_score(
                    y_test,
                    y_pred,
                    average="samples",
                    zero_division=0,
                )

                metrics = {
                    "attribute": attribute,
                    "task_type": "multi-label",
                    "hamming_loss": hamming,
                    "jaccard_micro": jaccard_micro,
                    "jaccard_macro": jaccard_macro,
                    "jaccard_samples": jaccard_samples,
                    "precision_macro": precision_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "recall_macro": recall_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "f1_macro": f1_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                }
                logger.info(
                    f"Hamming Loss: {hamming:.4f}, Jaccard (samples): {jaccard_samples:.4f}, "
                    f"F1 (macro): {metrics['f1_macro']:.4f}"
                )
        elif is_regression:
            if bootstrap_n_iterations > 0:
                logger.info(
                    f"Computing bootstrap confidence intervals ({bootstrap_n_iterations} iterations)..."
                )
                bootstrap_metrics = compute_regression_metrics_with_bootstrap(
                    y_test,
                    y_pred,
                    n_iterations=bootstrap_n_iterations,
                    random_state=random_state,
                )

                metrics = {
                    "attribute": attribute,
                    "task_type": "regression",
                }

                # Add mean values and confidence intervals
                for metric_name, (
                    mean_val,
                    lower_ci,
                    upper_ci,
                ) in bootstrap_metrics.items():
                    metrics[metric_name] = mean_val
                    metrics[f"{metric_name}_ci_lower"] = lower_ci
                    metrics[f"{metric_name}_ci_upper"] = upper_ci

                logger.info(f"MAE: {format_metric_with_ci(*bootstrap_metrics['mae'])}")
                logger.info(
                    f"RMSE: {format_metric_with_ci(*bootstrap_metrics['rmse'])}"
                )
                logger.info(f"R2: {format_metric_with_ci(*bootstrap_metrics['r2'])}")
            else:
                metrics = {
                    "attribute": attribute,
                    "task_type": "regression",
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred),
                }
                logger.info(
                    f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}"
                )
        else:
            if bootstrap_n_iterations > 0:
                logger.info(
                    f"Computing bootstrap confidence intervals ({bootstrap_n_iterations} iterations)..."
                )
                bootstrap_metrics = compute_classification_metrics_with_bootstrap(
                    y_test,
                    y_pred,
                    n_iterations=bootstrap_n_iterations,
                    random_state=random_state,
                    is_multilabel=False,
                )

                metrics = {
                    "attribute": attribute,
                    "task_type": "single-label",
                }

                # Add mean values and confidence intervals
                for metric_name, (
                    mean_val,
                    lower_ci,
                    upper_ci,
                ) in bootstrap_metrics.items():
                    metrics[metric_name] = mean_val
                    metrics[f"{metric_name}_ci_lower"] = lower_ci
                    metrics[f"{metric_name}_ci_upper"] = upper_ci

                logger.info(
                    f"Accuracy: {format_metric_with_ci(*bootstrap_metrics['accuracy'])}"
                )
                logger.info(
                    f"Balanced Acc: {format_metric_with_ci(*bootstrap_metrics['balanced_accuracy'])}"
                )
                logger.info(
                    f"F1 (macro): {format_metric_with_ci(*bootstrap_metrics['f1_macro'])}"
                )
            else:
                metrics = {
                    "attribute": attribute,
                    "task_type": "single-label",
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                }
                logger.info(
                    f"Accuracy: {metrics['accuracy']:.4f}, "
                    f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                    f"F1: {metrics['f1_macro']:.4f}"
                )

            # Compute confusion matrix for single-label classification
            cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
            all_confusion_matrices[attribute] = {
                "confusion_matrix": cm,
                "label_encoder": le,
            }
            logger.info(f"Computed confusion matrix for {attribute}")

        all_metrics.append(metrics)

        # Train final model on all known data (same configuration as evaluation)
        logger.info("Training final model on all known data...")
        model_final = _create_predictor_model(
            is_multilabel, is_regression, random_state
        )
        model_final.fit(X_known, y_known_encoded)

        # Initialize _pred column with ground truth where available
        df_output[f"{attribute}_pred"] = df_output[attribute].copy()

        # Predict missing values
        X_missing = embeddings[missing_value]
        logger.info(f"Predicting {len(X_missing)} missing values...")
        y_missing_pred = model_final.predict(X_missing)

        # Decode and fill predictions
        if is_multilabel:
            y_missing_pred_decoded = mlb.inverse_transform(y_missing_pred)
            y_missing_pred_decoded = [
                list(labels) if labels else [] for labels in y_missing_pred_decoded
            ]
        elif not is_regression:
            y_missing_pred_decoded = le.inverse_transform(y_missing_pred.astype(int))
        else:
            y_missing_pred_decoded = np.clip(y_missing_pred, clip_min, clip_max)
            logger.info(
                f"Clipped {len(y_missing_pred)} predictions to "
                f"range [{clip_min:.1f}, {clip_max:.1f}]"
            )

        # Fill in predictions for missing values only
        if is_multilabel:
            df_output.loc[missing_value, f"{attribute}_pred"] = np.array(
                y_missing_pred_decoded, dtype=object
            )
        else:
            df_output.loc[missing_value, f"{attribute}_pred"] = y_missing_pred_decoded

        # Save model if output directory provided
        if model_dir:
            model_path = os.path.join(model_dir, f"model_{attribute}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_final, f)
            logger.info(f"Saved model to {model_path}")

            # Save encoder
            if is_multilabel:
                mlb_path = os.path.join(model_dir, f"mlb_{attribute}.pkl")
                with open(mlb_path, "wb") as f:
                    pickle.dump(mlb, f)
                logger.info(f"Saved MultiLabelBinarizer to {mlb_path}")
            elif not is_regression:
                le_path = os.path.join(model_dir, f"label_encoder_{attribute}.pkl")
                with open(le_path, "wb") as f:
                    pickle.dump(le, f)
                logger.info(f"Saved LabelEncoder to {le_path}")

    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    return df_output, metrics_df, all_confusion_matrices


def predict_random_baseline_metadata(
    df: pd.DataFrame,
    attributes: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predict metadata using random baseline (for comparison).

    Random baseline predictions:
    - Single-label classification: randomly sample from class distribution
    - Multi-label classification: randomly sample from label distributions
    - Regression (age): randomly sample from value distribution

    Args:
        df: Metadata dataframe with N rows
        attributes: List of attribute column names to predict
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        output_dir: Directory to save trained models (optional, not used for random)

    Returns:
        Tuple of (updated dataframe with predictions, metrics dataframe, confusion matrices dict)
    """
    all_metrics = []
    all_confusion_matrices = {}
    df_output = df.copy()

    np.random.seed(random_state)

    for attribute in attributes:
        if attribute not in df.columns:
            logger.warning(f"Attribute {attribute} not found in dataframe, skipping")
            continue

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing attribute (RANDOM BASELINE): {attribute}")

        # Identify samples with known and missing values
        has_value = df[attribute].notna()
        missing_value = df[attribute].isna()

        n_has_value = has_value.sum()
        n_missing = missing_value.sum()

        logger.info(f"Samples with {attribute}: {n_has_value}")
        logger.info(f"Samples missing {attribute}: {n_missing}")

        if n_has_value < 10:
            logger.warning(
                f"Too few samples with {attribute} ({n_has_value}), skipping"
            )
            continue

        if n_missing == 0:
            logger.info(f"No missing values for {attribute}, skipping")
            continue

        # Determine task type
        is_regression = attribute == "age"

        # Check if this is multi-label classification
        is_multilabel = False
        if not is_regression:
            sample_values = df.loc[has_value, attribute].head(10)
            is_multilabel = any(isinstance(val, list) for val in sample_values)

        if is_multilabel:
            logger.info("Task type: Multi-label Classification (RANDOM)")
        elif is_regression:
            logger.info("Task type: Regression (RANDOM)")
        else:
            logger.info("Task type: Single-label Classification (RANDOM)")

        # Get labels for samples with known values
        y_known = df.loc[has_value, attribute].values

        # Encode labels based on task type
        mlb = None
        le = None

        if is_multilabel:
            mlb = MultiLabelBinarizer()
            y_known_encoded = mlb.fit_transform(y_known)
            n_labels = len(mlb.classes_)
            logger.info(f"Number of labels: {n_labels}")
        elif not is_regression:
            le = LabelEncoder()
            y_known_encoded = le.fit_transform(y_known)
            n_classes = len(le.classes_)
            logger.info(f"Number of classes: {n_classes}")
        else:
            y_known_encoded = y_known.astype(float)

        # Split for evaluation
        if is_multilabel:
            _, y_test_indices = train_test_split(
                np.arange(len(y_known_encoded)),
                test_size=test_size,
                random_state=random_state,
            )
        else:
            _, y_test_indices = train_test_split(
                np.arange(len(y_known_encoded)),
                test_size=test_size,
                random_state=random_state,
                stratify=y_known_encoded if not is_regression else None,
            )

        y_test = y_known_encoded[y_test_indices]
        logger.info(f"Test samples: {len(y_test)}")

        # Generate random predictions
        if is_multilabel:
            # For multi-label: randomly sample label combinations based on observed frequencies
            # Calculate frequency of each label appearing
            label_frequencies = y_known_encoded.mean(axis=0)

            # Generate random predictions by sampling each label independently
            y_pred = (
                np.random.rand(len(y_test), len(label_frequencies)) < label_frequencies
            )
            y_pred = y_pred.astype(int)
        elif is_regression:
            # For regression: sample from observed value distribution
            y_pred = np.random.choice(y_known_encoded, size=len(y_test), replace=True)
        else:
            # For single-label: sample from class distribution
            class_counts = np.bincount(y_known_encoded)
            class_probs = class_counts / class_counts.sum()
            y_pred = np.random.choice(len(class_probs), size=len(y_test), p=class_probs)

        logger.info("Generated random predictions for evaluation")

        # Calculate metrics based on task type
        if is_multilabel:
            hamming = hamming_loss(y_test, y_pred)
            jaccard_micro = jaccard_score(
                y_test,
                y_pred,
                average="micro",
                zero_division=0,
            )
            jaccard_macro = jaccard_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
            jaccard_samples = jaccard_score(
                y_test,
                y_pred,
                average="samples",
                zero_division=0,
            )

            metrics = {
                "attribute": attribute,
                "task_type": "multi-label",
                "hamming_loss": hamming,
                "jaccard_micro": jaccard_micro,
                "jaccard_macro": jaccard_macro,
                "jaccard_samples": jaccard_samples,
                "precision_macro": precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "recall_macro": recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1_macro": f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
            }
            logger.info(
                f"Hamming Loss: {hamming:.4f}, Jaccard (samples): {jaccard_samples:.4f}, "
                f"F1 (macro): {metrics['f1_macro']:.4f}"
            )
        elif is_regression:
            metrics = {
                "attribute": attribute,
                "task_type": "regression",
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }
            logger.info(
                f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}"
            )
        else:
            metrics = {
                "attribute": attribute,
                "task_type": "single-label",
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
            logger.info(
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                f"F1: {metrics['f1_macro']:.4f}"
            )

            # Compute confusion matrix for single-label classification
            cm = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
            all_confusion_matrices[attribute] = {
                "confusion_matrix": cm,
                "label_encoder": le,
            }
            logger.info(f"Computed confusion matrix for {attribute}")

        all_metrics.append(metrics)

        # Initialize _pred column with ground truth where available
        df_output[f"{attribute}_pred"] = df_output[attribute].copy()

        # Generate random predictions for missing values
        if n_missing > 0:
            logger.info(
                f"Generating random predictions for {n_missing} missing values..."
            )

            if is_multilabel:
                # Random multi-label predictions
                label_frequencies = y_known_encoded.mean(axis=0)
                y_missing_pred = (
                    np.random.rand(n_missing, len(label_frequencies))
                    < label_frequencies
                )
                y_missing_pred = y_missing_pred.astype(int)
                y_missing_pred_decoded = mlb.inverse_transform(y_missing_pred)
                y_missing_pred_decoded = [
                    list(labels) if labels else [] for labels in y_missing_pred_decoded
                ]
            elif not is_regression:
                # Random single-label predictions
                class_counts = np.bincount(y_known_encoded)
                class_probs = class_counts / class_counts.sum()
                y_missing_pred = np.random.choice(
                    len(class_probs), size=n_missing, p=class_probs
                )
                y_missing_pred_decoded = le.inverse_transform(
                    y_missing_pred.astype(int)
                )
            else:
                # Random regression predictions
                y_missing_pred_decoded = np.random.choice(
                    y_known_encoded, size=n_missing, replace=True
                )

            # Fill in predictions for missing values only
            if is_multilabel:
                df_output.loc[missing_value, f"{attribute}_pred"] = np.array(
                    y_missing_pred_decoded, dtype=object
                )
            else:
                df_output.loc[missing_value, f"{attribute}_pred"] = (
                    y_missing_pred_decoded
                )

    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    return df_output, metrics_df, all_confusion_matrices


def evaluate_metadata_with_balancing_comparison(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    attributes: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Compare metadata prediction with balanced vs imbalanced training data.

    NOTE: Under/oversampling only applies to CLASSIFICATION tasks.
    Regression tasks (continuous variables) are automatically skipped.

    Args:
        embeddings: Image embeddings (N, D)
        df: Metadata dataframe
        attributes: Attributes to predict (only classification tasks are evaluated)
        test_size: Test split proportion
        random_state: Random seed
        output_dir: Output directory

    Returns:
        DataFrame with comparison results (only classification tasks)
    """
    results = []

    for attr in attributes:
        if attr not in df.columns or df[attr].notna().sum() < 20:
            continue

        # Skip regression and multi-label tasks (balancing only works for single-label classification)
        has_val = df[attr].notna()
        is_regression = (
            attr == "age"
        )  # Only age is treated as regression (matches existing logic)
        is_multilabel = any(isinstance(v, list) for v in df.loc[has_val, attr].head(10))

        if is_regression or is_multilabel:
            logger.debug(
                f"Skipping {attr} ({'regression' if is_regression else 'multi-label'})"
            )
            continue

        # Encode and split
        X = embeddings[has_val]
        le = LabelEncoder()
        y = le.fit_transform(df.loc[has_val, attr].values)

        # Skip if only one class (can't train classifier)
        if len(np.unique(y)) < 2:
            logger.debug(f"Skipping {attr} (only one class)")
            continue

        # Skip if any class has only 1 sample (stratified split needs at least 2 per class)
        class_counts = np.bincount(y)
        if np.any(class_counts < 2):
            logger.debug(f"Skipping {attr} (class with only 1 sample, can't stratify)")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Evaluate each strategy
        for strategy in ["none", "undersample", "oversample"]:
            X_bal, y_bal = (
                balance_dataset(X_tr, y_tr, strategy, random_state)
                if strategy != "none"
                else (X_tr, y_tr)
            )
            clf = _create_predictor_model(
                is_multilabel=False, is_regression=False, random_state=random_state
            )
            clf.fit(X_bal, y_bal)
            y_pred = clf.predict(X_te)

            results.append(
                {
                    "attribute": attr,
                    "strategy": strategy,
                    "accuracy": accuracy_score(y_te, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
                    "f1_macro": f1_score(
                        y_te, y_pred, average="macro", zero_division=0
                    ),
                }
            )

    results_df = pd.DataFrame(results)
    if output_dir and not results_df.empty:
        os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
        results_df.to_csv(
            os.path.join(output_dir, "analysis", "metadata_balancing_comparison.csv"),
            index=False,
        )
    return results_df

"""Downstream evaluation tasks for embeddings.

Evaluates embeddings on classification and regression tasks using various classifiers.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ..data.transforms import get_imagenet_transform
from ..models.loaders import normalize_model_tuple
from ..utils.balancing import balance_dataset
from ..utils.bootstrap import (
    compute_classification_metrics_with_bootstrap,
    compute_regression_metrics_with_bootstrap,
    format_metric_with_ci,
)
from ..utils.metadata import _create_predictor_model


def _evaluate_classification_task(
    X: np.ndarray,
    dataset,
    col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    classifier_names: list = ["linear", "knn10", "knn50", "xgboost"],
    learned_metric_L: Optional[np.ndarray] = None,
    return_results: bool = False,
    use_random_baseline: bool = False,
    bootstrap_n_iterations: int = 0,
):
    """Helper function to evaluate a classification task.

    Args:
        X: Embeddings array
        dataset: Dataset with metadata
        col: Column name to evaluate
        test_size: Test split size
        random_state: Random seed
        classifier_names: List of classifier names
        learned_metric_L: Pre-computed metric transformation matrix (optional)
        return_results: If True, return results dictionary instead of just logging
        bootstrap_n_iterations: Number of bootstrap iterations for uncertainty estimation (0 = disabled)

    Returns:
        Tuple of (results_dict, confusion_matrices_dict, label_encoder) if return_results=True, otherwise (None, None, None)
    """
    collected_results = {} if return_results else None
    collected_confusion_matrices = {} if return_results else None

    # Map "UNK" to np.nan and filter out
    meta_data = dataset.meta_data.copy()
    meta_data.loc[meta_data[col] == "UNK", col] = np.nan

    # Filter by counts and encode labels
    counts = meta_data[col].value_counts()
    df_filtered = meta_data[meta_data[col].isin(counts[counts > 1].index)]

    if len(df_filtered) == 0:
        logger.warning(
            f"No valid values for classification column '{col}', skipping..."
        )
        return (
            (collected_results, collected_confusion_matrices, None)
            if return_results
            else (None, None, None)
        )

    labels = df_filtered[col].values
    embeddings = X[df_filtered.index]

    # Encode labels to consecutive integers for XGBoost compatibility
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_encoded,
    )

    # Evaluate in original space
    if use_random_baseline:
        logger.info("  [Random Baseline]")
    else:
        logger.info("  [Original Space]")
    for clf_name in classifier_names:
        if use_random_baseline:
            # Random baseline: predict based on class distribution
            class_counts = np.bincount(y_train)
            class_probs = class_counts / class_counts.sum()
            np.random.seed(random_state)
            y_pred = np.random.choice(len(class_probs), size=len(y_test), p=class_probs)
        else:
            if clf_name == "linear":
                clf = LogisticRegression(max_iter=5_000, random_state=random_state)
            elif clf_name == "knn10":
                clf = KNeighborsClassifier(n_neighbors=10, metric="cosine")
            elif clf_name == "knn50":
                clf = KNeighborsClassifier(n_neighbors=50, metric="cosine")
            elif clf_name == "xgboost":
                clf = xgb.XGBClassifier()
            else:
                raise ValueError(f"Unknown classifier: {clf_name}")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        if bootstrap_n_iterations > 0:
            # Compute bootstrap confidence intervals
            bootstrap_metrics = compute_classification_metrics_with_bootstrap(
                y_test,
                y_pred,
                n_iterations=bootstrap_n_iterations,
                random_state=random_state,
                is_multilabel=False,
            )

            results = {}
            for metric_name, (
                mean_val,
                lower_ci,
                upper_ci,
            ) in bootstrap_metrics.items():
                results[metric_name] = mean_val
                results[f"{metric_name}_ci_lower"] = lower_ci
                results[f"{metric_name}_ci_upper"] = upper_ci

            # Log key metrics: accuracy, balanced_accuracy, f1_macro
            logger.info(
                f"  {clf_name}: "
                f"Acc: {format_metric_with_ci(*bootstrap_metrics['accuracy'])}, "
                f"Bal Acc: {format_metric_with_ci(*bootstrap_metrics['balanced_accuracy'])}, "
                f"F1: {format_metric_with_ci(*bootstrap_metrics['f1_macro'])}"
            )
        else:
            results = {
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
            logger.info(f"  {clf_name}: {results}")

        if return_results:
            collected_results[clf_name] = results
            # Compute confusion matrix with all possible labels from encoder
            cm = confusion_matrix(
                y_test, y_pred, labels=range(len(label_encoder.classes_))
            )
            collected_confusion_matrices[clf_name] = cm

    # Evaluate KNN in learned metric space (skip for random baseline)
    if learned_metric_L is not None and not use_random_baseline:
        # Check dimension compatibility before applying transformation
        embedding_dim = X_train.shape[1]
        metric_dim = learned_metric_L.shape[1]

        if embedding_dim != metric_dim:
            logger.warning(
                f"Skipping learned metric evaluation: dimension mismatch "
                f"(embeddings: {embedding_dim}, metric: {metric_dim}). "
                f"This can happen when using truncated whitening on the main dataset "
                f"but extracting full embeddings for evaluation datasets."
            )
        else:
            logger.info("  [Learned Metric Space]")

            # Transform embeddings: X_transformed = X @ L.T
            X_train_transformed = X_train @ learned_metric_L.T
            X_test_transformed = X_test @ learned_metric_L.T

            for clf_name in classifier_names:
                if "knn" not in clf_name:
                    continue  # Only evaluate KNN in transformed space

                if clf_name == "knn10":
                    clf = KNeighborsClassifier(n_neighbors=10, metric="euclidean")
                elif clf_name == "knn50":
                    clf = KNeighborsClassifier(n_neighbors=50, metric="euclidean")

                clf.fit(X_train_transformed, y_train)
                y_pred = clf.predict(X_test_transformed)

                if bootstrap_n_iterations > 0:
                    # Compute bootstrap confidence intervals
                    bootstrap_metrics = compute_classification_metrics_with_bootstrap(
                        y_test,
                        y_pred,
                        n_iterations=bootstrap_n_iterations,
                        random_state=random_state,
                        is_multilabel=False,
                    )

                    results = {}
                    for metric_name, (
                        mean_val,
                        lower_ci,
                        upper_ci,
                    ) in bootstrap_metrics.items():
                        results[metric_name] = mean_val
                        results[f"{metric_name}_ci_lower"] = lower_ci
                        results[f"{metric_name}_ci_upper"] = upper_ci

                    # Log key metrics: accuracy, balanced_accuracy, f1_macro
                    logger.info(
                        f"  {clf_name}_learned: "
                        f"Acc: {format_metric_with_ci(*bootstrap_metrics['accuracy'])}, "
                        f"Bal Acc: {format_metric_with_ci(*bootstrap_metrics['balanced_accuracy'])}, "
                        f"F1: {format_metric_with_ci(*bootstrap_metrics['f1_macro'])}"
                    )
                else:
                    results = {
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
                    logger.info(f"  {clf_name}_learned: {results}")

                if return_results:
                    collected_results[f"{clf_name}_learned"] = results
                    # Compute confusion matrix with all possible labels from encoder
                    cm = confusion_matrix(
                        y_test, y_pred, labels=range(len(label_encoder.classes_))
                    )
                    collected_confusion_matrices[f"{clf_name}_learned"] = cm

    return (
        (collected_results, collected_confusion_matrices, label_encoder)
        if return_results
        else (None, None, None)
    )


def _evaluate_regression_task(
    X: np.ndarray,
    dataset,
    col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    regressor_names: list = ["linear", "knn10", "knn50", "xgboost"],
    return_results: bool = False,
    use_random_baseline: bool = False,
    bootstrap_n_iterations: int = 0,
):
    """Helper function to evaluate a regression task.

    Args:
        X: Embeddings array
        dataset: Dataset with metadata
        col: Column name to evaluate
        test_size: Test split size
        random_state: Random seed
        regressor_names: List of regressor names
        return_results: If True, return results dictionary instead of just logging
        bootstrap_n_iterations: Number of bootstrap iterations for uncertainty estimation (0 = disabled)

    Returns:
        Dictionary of results if return_results=True, otherwise None
    """
    collected_results = {} if return_results else None
    # Filter out NaN values and convert to float
    df_filtered = dataset.meta_data[dataset.meta_data[col].notna()]

    if len(df_filtered) == 0:
        logger.warning(f"No valid values for regression column '{col}', skipping...")
        return collected_results if return_results else None

    labels = df_filtered[col].astype(float).values
    embeddings = X[df_filtered.index]

    # No stratification for regression
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    # Train and evaluate regression models
    for reg_name in regressor_names:
        if use_random_baseline:
            # Random baseline: sample from training distribution
            np.random.seed(random_state)
            y_pred = np.random.choice(y_train, size=len(y_test), replace=True)
        else:
            if reg_name == "linear":
                from sklearn.linear_model import Ridge

                reg = Ridge(alpha=1.0, random_state=random_state)
            elif reg_name == "knn10":
                from sklearn.neighbors import KNeighborsRegressor

                reg = KNeighborsRegressor(n_neighbors=10, metric="cosine")
            elif reg_name == "knn50":
                from sklearn.neighbors import KNeighborsRegressor

                reg = KNeighborsRegressor(n_neighbors=50, metric="cosine")
            elif reg_name == "xgboost":
                reg = xgb.XGBRegressor()
            else:
                raise ValueError(f"Unknown regressor: {reg_name}")

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

        # Regression metrics
        if bootstrap_n_iterations > 0:
            # Compute bootstrap confidence intervals
            bootstrap_metrics = compute_regression_metrics_with_bootstrap(
                y_test,
                y_pred,
                n_iterations=bootstrap_n_iterations,
                random_state=random_state,
            )

            results = {}
            for metric_name, (
                mean_val,
                lower_ci,
                upper_ci,
            ) in bootstrap_metrics.items():
                results[metric_name] = mean_val
                results[f"{metric_name}_ci_lower"] = lower_ci
                results[f"{metric_name}_ci_upper"] = upper_ci

            # Log all regression metrics
            logger.info(
                f"  {reg_name}: "
                f"MAE: {format_metric_with_ci(*bootstrap_metrics['mae'])}, "
                f"RMSE: {format_metric_with_ci(*bootstrap_metrics['rmse'])}, "
                f"R2: {format_metric_with_ci(*bootstrap_metrics['r2'])}"
            )
        else:
            results = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }
            logger.info(f"--- {reg_name} ---")
            logger.info(f"{results}")

        if return_results:
            collected_results[reg_name] = results

    return collected_results if return_results else None


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def _apply_whiten(z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (z - mu) @ W.T


def _normalize_teacher_blocks(z_cat: np.ndarray, dims: List[int]) -> np.ndarray:
    """L2 normalize each teacher block in concatenated embeddings."""
    start = 0
    for dim in dims:
        block = z_cat[:, start : start + dim]
        z_cat[:, start : start + dim] = _l2_normalize(block, axis=1)
        start += dim
    return z_cat


def _get_projector_out_dim(projector_model) -> Optional[int]:
    """Best-effort output dimension introspection for projector image head."""
    if projector_model is None:
        return None
    img_head = getattr(projector_model, "img_head", None)
    if img_head is None:
        return None
    if hasattr(img_head, "proj"):
        return int(img_head.proj.out_features)
    if hasattr(img_head, "lin2"):
        return int(img_head.lin2.out_features)
    return None


def _project_embeddings_with_projector(
    all_image_embeddings: List[np.ndarray],
    projector_model,
    whitening_stats: Dict,
    skip_whitening: bool,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
) -> Optional[np.ndarray]:
    """Apply whitening + trained projector to per-model embeddings."""
    if projector_model is None or whitening_stats is None:
        return None

    dims = whitening_stats.get("dims", [])
    if not dims:
        logger.warning("Projector whitening stats missing dims; skipping projection")
        return None

    if len(all_image_embeddings) != len(dims):
        logger.warning(
            "Projector dims mismatch: %d embedding blocks vs %d whitening dims; "
            "skipping projection",
            len(all_image_embeddings),
            len(dims),
        )
        return None

    mu_list = whitening_stats.get("mu", [])
    W_list = whitening_stats.get("W", [])
    if not skip_whitening and (len(mu_list) != len(dims) or len(W_list) != len(dims)):
        logger.warning("Whitening stats incomplete; skipping projection")
        return None

    whitened_blocks = []
    for i, embs in enumerate(all_image_embeddings):
        embs_normed = _l2_normalize(embs, axis=1)
        if not skip_whitening:
            embs_normed = _apply_whiten(embs_normed, mu_list[i], W_list[i])
        whitened_blocks.append(embs_normed.astype(np.float32, copy=False))

    z_cat = np.concatenate(whitened_blocks, axis=1)
    if z_cat.shape[1] != int(sum(dims)):
        logger.warning(
            "Concatenated dims %d do not match whitening dims sum %d; "
            "skipping projection",
            z_cat.shape[1],
            int(sum(dims)),
        )
        return None

    z_cat = _normalize_teacher_blocks(z_cat, dims)

    projector_model.eval()
    if device is None:
        device = next(projector_model.parameters()).device
    projector_model.to(device)

    projected = []
    with torch.no_grad():
        for start in range(0, len(z_cat), batch_size):
            end = min(start + batch_size, len(z_cat))
            z_cat_batch = torch.from_numpy(z_cat[start:end]).float().to(device)
            zi = projector_model.img_head(z_cat_batch)
            projected.append(zi.cpu().numpy())

    return np.vstack(projected)


def evaluate_image_dataset(
    model,
    processor,
    device,
    dataset,
    classification_cols: Optional[list] = None,
    regression_cols: Optional[list] = None,
    emb_path: Optional[str] = None,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 4,
    models_and_processors=None,
    svd_components: Optional[int] = None,
    is_ssl_model: bool = False,
    classifier_names: list = ["linear", "knn10", "knn50", "xgboost"],
    learned_metric_L: Optional[np.ndarray] = None,
    projector_model=None,
    projector_whitening_stats: Optional[Dict] = None,
    projector_skip_whitening: bool = False,
    return_results: bool = False,
    use_random_baseline: bool = False,
    bootstrap_n_iterations: int = 0,
):
    """
    Extracts embeddings for an image-only dataset and evaluates classification/regression tasks.

    Args:
        model: CLIP-compatible model with get_image_features method.
        processor: HuggingFace processor for images.
        device: torch.device to run inference on.
        dataset: a torch Dataset returning (PIL.Image, label).
        classification_cols: List of metadata columns for classification tasks.
        regression_cols: List of metadata columns for regression tasks.
        emb_path: Path to save/load embeddings.
        batch_size: batch size for embedding extraction.
        test_size: proportion of data to use as test set.
        random_state: seed for reproducibility.
        num_workers: number of DataLoader workers.
        models_and_processors: Optional list of (model, processor, model_type) tuples for ensemble.
        svd_components: Optional number of SVD components for dimensionality reduction.
        is_ssl_model: Whether using SSL model.
        classifier_names: List of classifier/regressor names to use.
        return_results: If True, return dictionary of all results instead of just logging.
        use_random_baseline: If True, use random baseline instead of trained classifiers.
        bootstrap_n_iterations: Number of bootstrap iterations for uncertainty estimation (0 = disabled).
        projector_model: Optional trained projector for embedding projection.
        projector_whitening_stats: Whitening stats for projector preprocessing.
        projector_skip_whitening: Skip whitening before projector if True.

    Returns:
        Tuple of (results, confusion_matrices) if return_results=True, otherwise (None, None).
        results: Dictionary with structure {task_type: {task_name: {classifier: metrics}}}
        confusion_matrices: Dictionary with structure {task_name: {classifier: (confusion_matrix, label_encoder)}}
    """
    all_results = {"classification": {}, "regression": {}} if return_results else None
    all_confusion_matrices = {} if return_results else None

    expected_projector_dim = _get_projector_out_dim(projector_model)

    # Prepare DataLoader with a custom collate_fn
    def collate_fn(batch):
        images, _, labels = zip(*batch)
        # Processor handles list of PIL Images
        inputs = processor(images=list(images), return_tensors="pt")
        pixel_values = inputs.pixel_values
        return pixel_values, labels

    # Import at function level to avoid scope issues
    from torch.utils.data import DataLoader

    recompute_embeddings = False
    if emb_path and os.path.exists(emb_path):
        with np.load(emb_path, allow_pickle=True) as data:
            X = data["image_embeddings"]
            y = data["labels"]
        if expected_projector_dim is not None and X.shape[1] != expected_projector_dim:
            logger.info(
                "Cached embeddings dim %d != projector dim %d; recomputing embeddings",
                X.shape[1],
                expected_projector_dim,
            )
            recompute_embeddings = True

    if not emb_path or not os.path.exists(emb_path) or recompute_embeddings:
        if is_ssl_model:
            # SSL model case - use embed_dataset directly
            logger.info("Using SSL model for embedding extraction")

            transform = get_imagenet_transform()

            # Create a simple dataset wrapper that applies transforms
            class TransformDataset(torch.utils.data.Dataset):
                def __init__(self, original_dataset, transform):
                    self.dataset = original_dataset
                    self.transform = transform

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    image, path, label = self.dataset[idx]
                    if self.transform:
                        image = self.transform(image)
                    return image, label

            transform_dataset = TransformDataset(dataset, transform)
            dataloader = DataLoader(
                transform_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            # Extract embeddings manually for SSL models since they don't use n_layers parameter
            model.eval()
            all_embeddings = []
            all_labels = []

            with torch.no_grad():
                for batch_images, batch_labels in tqdm(
                    dataloader, desc="SSL embedding extraction"
                ):
                    batch_images = batch_images.to(device)
                    # SSL models are called directly without n_layers parameter
                    emb = model(batch_images)
                    # Normalize embeddings
                    emb = torch.nn.functional.normalize(emb, dim=-1, p=2)
                    all_embeddings.append(emb.cpu())
                    all_labels.append(batch_labels)

            # Concatenate all embeddings
            X = torch.cat(all_embeddings, dim=0).numpy()
            y = torch.cat(all_labels, dim=0).numpy()
            all_image_embeddings = [X]

        elif models_and_processors is not None:
            # Use multiple models
            logger.info("Using combined models for embedding extraction")
            all_image_embeddings = []

            for i, model_info in enumerate(models_and_processors):
                m, p, model_type = normalize_model_tuple(model_info)

                logger.info(
                    f"Extracting embeddings from model {i+1}/{len(models_and_processors)} (type: {model_type})"
                )

                if model_type == "ssl":
                    # SSL models need different handling
                    transform = get_imagenet_transform()

                    # Create SSL-compatible dataset
                    class SSLTransformDataset(torch.utils.data.Dataset):
                        def __init__(self, original_dataset, transform):
                            self.dataset = original_dataset
                            self.transform = transform

                        def __len__(self):
                            return len(self.dataset)

                        def __getitem__(self, idx):
                            image, path, label = self.dataset[idx]
                            if self.transform:
                                image = self.transform(image)
                            return image, label

                    ssl_dataset = SSLTransformDataset(dataset, transform)
                    loader = DataLoader(
                        ssl_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                    )

                    m.eval()
                    model_embeddings = []
                    model_labels = []

                    with torch.no_grad():
                        for batch_images, labels in tqdm(
                            loader, desc=f"SSL Model {i+1} embedding samples"
                        ):
                            batch_images = batch_images.to(device)
                            img_emb = m(batch_images)  # SSL models called directly
                            img_emb = torch.nn.functional.normalize(
                                img_emb, dim=-1, p=2
                            )
                            model_embeddings.append(img_emb.cpu().numpy())
                            if i == 0:  # Only collect labels once
                                model_labels.extend(labels)
                else:
                    # CLIP models - create collate_fn for this specific processor
                    def clip_collate_fn(batch):
                        images, _, labels = zip(*batch)
                        # Use the processor for this specific model
                        inputs = p(images=list(images), return_tensors="pt")
                        pixel_values = inputs.pixel_values
                        return pixel_values, labels

                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=clip_collate_fn,
                    )

                    m.eval()
                    model_embeddings = []
                    model_labels = []

                    with torch.no_grad():
                        for pixel_values, labels in tqdm(
                            loader, desc=f"CLIP Model {i+1} embedding samples"
                        ):
                            pixel_values = pixel_values.to(device)
                            img_emb = m.get_image_features(pixel_values)
                            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                            model_embeddings.append(img_emb.cpu().numpy())
                            if i == 0:  # Only collect labels once
                                model_labels.extend(labels)

                all_image_embeddings.append(np.vstack(model_embeddings))
                if i == 0:
                    y = np.array(model_labels)

            # Concatenate embeddings from all models
            X = np.concatenate(all_image_embeddings, axis=1)
            logger.info(f"Combined embedding dimensions: {X.shape[1]}")
        else:
            # Use single model
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            # Switch model to eval mode
            model.eval()
            all_embeddings = []
            all_labels = []

            with torch.no_grad():
                for pixel_values, labels in tqdm(loader, desc="Embedding samples"):
                    pixel_values = pixel_values.to(device)
                    # Extract image embeddings
                    img_emb = model.get_image_features(pixel_values)
                    # Optionally normalize
                    img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
                    # Move to CPU and convert to numpy
                    all_embeddings.append(img_emb.cpu().numpy())
                    all_labels.extend(labels)

            # Concatenate
            X = np.vstack(all_embeddings)
            y = np.array(all_labels)
            all_image_embeddings = [X]

        # Apply projector if provided
        if projector_model is not None and projector_whitening_stats is not None:
            if svd_components is not None:
                logger.info("Projector provided; ignoring svd_components for eval")
            projected = _project_embeddings_with_projector(
                all_image_embeddings,
                projector_model,
                projector_whitening_stats,
                projector_skip_whitening,
                device=device,
            )
            if projected is not None:
                X = projected
                logger.info(f"Projected embedding dimensions: {X.shape[1]}")
        elif svd_components is not None:
            logger.info(f"Applying truncated SVD with {svd_components} components")
            svd = TruncatedSVD(n_components=svd_components, random_state=42)
            X = svd.fit_transform(X)
            logger.info(f"Reduced embedding dimensions to: {X.shape[1]}")

        # Save embeddings
        if emb_path:
            Path(emb_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                emb_path,
                image_embeddings=X,
                labels=y,
            )
            logger.info(f"Saved embeddings to {emb_path}")

    # Evaluate classification tasks
    if classification_cols:
        logger.info(f"Evaluating {len(classification_cols)} classification tasks")
        for col in classification_cols:
            logger.info(f"\n=== Classification: {col} ===")
            col_results, col_confusion_matrices, col_label_encoder = (
                _evaluate_classification_task(
                    X,
                    dataset,
                    col,
                    test_size,
                    random_state,
                    classifier_names,
                    learned_metric_L,
                    return_results=return_results,
                    use_random_baseline=use_random_baseline,
                    bootstrap_n_iterations=bootstrap_n_iterations,
                )
            )
            if return_results and col_results:
                all_results["classification"][col] = col_results
                # Store confusion matrices with label encoder for later decoding
                all_confusion_matrices[col] = {
                    "confusion_matrices": col_confusion_matrices,
                    "label_encoder": col_label_encoder,
                }

    # Evaluate regression tasks
    if regression_cols:
        logger.info(f"Evaluating {len(regression_cols)} regression tasks")
        for col in regression_cols:
            logger.info(f"\n=== Regression: {col} ===")
            col_results = _evaluate_regression_task(
                X,
                dataset,
                col,
                test_size,
                random_state,
                regressor_names=classifier_names,
                return_results=return_results,
                use_random_baseline=use_random_baseline,
                bootstrap_n_iterations=bootstrap_n_iterations,
            )
            if return_results and col_results:
                all_results["regression"][col] = col_results

    return (all_results, all_confusion_matrices) if return_results else (None, None)


def evaluate_downstream_with_balancing_comparison(
    embeddings: np.ndarray,
    dataset,
    classification_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Compare downstream tasks with balanced vs imbalanced training data.

    NOTE: Under/oversampling only applies to CLASSIFICATION tasks.
    This function only accepts classification columns (no regression tasks).

    Args:
        embeddings: Pre-computed embeddings (N, D)
        dataset: Dataset with .meta_data
        classification_cols: Classification columns only (no continuous/regression variables)
        test_size: Test split proportion
        random_state: Random seed
        output_dir: Output directory
        dataset_name: Dataset name for saving

    Returns:
        DataFrame with comparison results
    """
    results = []

    for col in classification_cols or []:
        # Filter valid data
        meta = dataset.meta_data.copy()
        meta.loc[meta[col] == "UNK", col] = np.nan
        counts = meta[col].value_counts()
        df_filt = meta[meta[col].isin(counts[counts > 1].index)]

        if len(df_filt) == 0:
            continue

        # Encode and split
        X = embeddings[df_filt.index]
        le = LabelEncoder()
        y = le.fit_transform(df_filt[col].values)

        # Skip if only one class (can't train classifier)
        if len(np.unique(y)) < 2:
            logger.debug(f"Skipping {col} (only one class)")
            continue

        # Skip if any class has only 1 sample (stratified split needs at least 2 per class)
        class_counts = np.bincount(y)
        if np.any(class_counts < 2):
            logger.debug(f"Skipping {col} (class with only 1 sample, can't stratify)")
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
                    "dataset": dataset_name,
                    "task": col,
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
            os.path.join(
                output_dir, "analysis", f"downstream_balancing_{dataset_name}.csv"
            ),
            index=False,
        )
    return results_df

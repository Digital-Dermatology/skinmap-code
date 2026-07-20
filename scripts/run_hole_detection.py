#!/usr/bin/env python3
"""CLI to run N-dimensional hole detection on synthetic or real data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hole_detection import (
    EffectiveResistancePHDetector,
    compute_metrics,
    format_metrics,
    synthetic,
)

_LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def add_detector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eff-backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="Backend for effective-resistance kNN.",
    )
    parser.add_argument(
        "--torch-device",
        help="Torch device to use when eff-backend=torch (e.g., cuda or cuda:0).",
    )
    parser.add_argument(
        "--torch-chunk-size",
        type=int,
        default=8_192,
        help="Query batch size for torch kNN accumulation.",
    )
    parser.add_argument(
        "--eff-k",
        type=int,
        default=30,
        help="k for effective-resistance PH detector.",
    )
    parser.add_argument(
        "--eff-corrected",
        action="store_true",
        help="Apply von Luxburg degree correction in effective resistance.",
    )
    parser.add_argument(
        "--eff-weighted",
        action="store_true",
        help="Use weighted kNN graph for effective resistance.",
    )
    parser.add_argument(
        "--eff-min-persistence",
        type=float,
        default=0.0,
        help="Minimum H1 persistence to keep a hole.",
    )
    parser.add_argument(
        "--eff-max-holes",
        type=int,
        help="Optional cap on number of holes to keep (by persistence).",
    )
    parser.add_argument(
        "--eff-input-metric",
        type=str,
        default="cosine",
        help="Metric for kNN when using effective resistance.",
    )
    parser.add_argument(
        "--eff-boundary-neighbor-k",
        type=int,
        default=25,
        help="Neighbors used to sample boundary points around cocycle nodes.",
    )
    parser.add_argument(
        "--eff-boundary-radius-scale",
        type=float,
        default=1.5,
        help="Scale on median NN distance for boundary sampling radius.",
    )
    parser.add_argument(
        "--eff-landmarks",
        type=int,
        help="If set, use this many landmarks (random/fps) for ER + PH.",
    )
    parser.add_argument(
        "--eff-landmark-method",
        choices=["random", "fps"],
        default="random",
        help="Landmark selection strategy.",
    )
    parser.add_argument(
        "--eff-use-witness-complex",
        action="store_true",
        help="Use a witness/landmark complex for PH (leverages non-landmark points).",
    )
    parser.add_argument(
        "--eff-witness-k",
        type=int,
        default=5,
        help="Number of nearest landmarks each witness contributes to (strong witness).",
    )
    parser.add_argument(
        "--eff-witness-batch",
        type=int,
        default=5000,
        help="Batch size when querying witness->landmark neighbors.",
    )
    parser.add_argument(
        "--eff-subsample",
        type=int,
        help="Optional random subsample size for ER-PH fit (applied before landmarks).",
    )
    parser.add_argument(
        "--eff-save-diagram",
        action="store_true",
        help="Save H1 diagram and cocycle info to NPZ alongside outputs for debugging.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        help="Path to save detailed hole summary (CSV). Defaults to <input>.holes.csv.",
    )
    parser.add_argument(
        "--nn-neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors per hole to report in the summary CSV.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        help="Optional PCA components to project data before detection (e.g., 64 or 128).",
    )
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Whiten PCA output (scales components to unit variance).",
    )
    parser.add_argument(
        "--pca-fit-samples",
        type=int,
        default=20000,
        help="Samples used to fit PCA (subset for efficiency).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=_LOG_LEVELS,
        help="Verbosity for loguru output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    syn = sub.add_parser("synthetic", help="Run synthetic experiments.")
    syn.add_argument(
        "--datasets",
        nargs="+",
        default=["shell"],
        help="Synthetic dataset types to evaluate.",
    )
    syn.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2, 3, 5],
        help="Dimensionalities to evaluate.",
    )
    syn.add_argument(
        "--samples",
        type=int,
        default=4000,
        help="Number of samples per synthetic dataset.",
    )
    syn.add_argument(
        "--visualize",
        action="store_true",
        help="Placeholder for future visualisations (no effect).",
    )
    add_detector_args(syn)

    app = sub.add_parser("apply", help="Run detector on an input dataset.")
    app.add_argument(
        "--input",
        required=True,
        help="Path to dataset (csv, parquet, npy, npz).",
    )
    app.add_argument(
        "--columns",
        nargs="+",
        help="Subset of columns (for tabular files).",
    )
    app.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter.",
    )
    app.add_argument(
        "--output",
        help="Optional path to save detected hole centers (CSV).",
    )
    add_detector_args(app)

    return parser


def synthetic_experiments(args: argparse.Namespace) -> None:
    dims = sorted(set(args.dimensions))
    datasets = [name.lower() for name in args.datasets]

    results: List[Dict[str, float]] = []

    detector_kwargs = dict(
        k=args.eff_k,
        corrected=args.eff_corrected,
        weighted=args.eff_weighted,
        backend=args.eff_backend,
        input_metric=args.eff_input_metric,
        min_persistence=args.eff_min_persistence,
        max_holes=args.eff_max_holes,
        torch_chunk_size=args.torch_chunk_size,
        torch_device=args.torch_device,
        boundary_neighbor_k=args.eff_boundary_neighbor_k,
        boundary_radius_scale=args.eff_boundary_radius_scale,
        landmark_count=args.eff_landmarks,
        landmark_method=args.eff_landmark_method,
        use_witness_complex=args.eff_use_witness_complex,
        witness_k=args.eff_witness_k,
        witness_batch_size=args.eff_witness_batch,
    )

    for dataset in datasets:
        for d in dims:
            X, ground_truth = generate_dataset(dataset, args.samples, d, args.seed)
            detector = EffectiveResistancePHDetector(**detector_kwargs)
            detector.fit(X)
            detector.detect_holes()
            metrics = compute_metrics(detector, ground_truth)
            metrics["dataset"] = dataset
            metrics["dimension"] = d
            results.append(metrics)

            logger.info("Dataset={} | dim={}", dataset, d)
            logger.info("\n{}", format_metrics(metrics))

    if results:
        df = pd.DataFrame(results)
        logger.info("Aggregated results (csv/json can be saved as needed):")
        summary = df.reindex(
            columns=[
                "dataset",
                "dimension",
                "n_detected_holes",
                "precision",
                "recall",
                "f1_score",
            ]
        ).fillna(0.0)
        logger.info("\n{}", summary.to_string(index=False))


def generate_dataset(
    dataset: str, n: int, d: int, seed: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    dataset = dataset.lower()

    if dataset == "shell":
        X = synthetic.sample_hypersphere_shell(n, d, seed=seed)
        ground_truth = np.zeros((1, d))
    elif dataset == "c_shape":
        X = synthetic.sample_c_shape_nd(n, d, seed=seed)
        ground_truth = None
    elif dataset == "two_shells":
        X = synthetic.sample_two_shells(n, d, seed=seed)
        gt = np.zeros((3, d))
        gt[1, 0] = 6.0
        gt[2, 0] = 3.0
        ground_truth = gt
    elif dataset == "two_rings_small":
        X = synthetic.sample_two_rings_small(n, d, seed=seed)
        gt = np.zeros((2, d))
        gt[1, 0] = 3.0
        ground_truth = gt
    elif dataset == "shell_with_blob":
        X = synthetic.sample_shell_with_blob(n, d, seed=seed)
        ground_truth = np.zeros((1, d))
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

    return X, ground_truth


def load_data(
    path: Path, columns: Optional[Sequence[str]], delimiter: str
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".npy"}:
        data = np.load(path)
        return np.asarray(data, dtype=float)
    if suffix in {".npz"}:
        archive = np.load(path)
        key = archive.files[0]
        return np.asarray(archive[key], dtype=float)
    if suffix in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=delimiter)

    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in {path}: {missing}")
        df = df[list(columns)]
    else:
        df = df.select_dtypes(include=[np.number])

    if df.empty:
        raise ValueError("No numeric columns available for detection.")
    return df.to_numpy(dtype=float)


def apply_detector(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    detector_kwargs = dict(
        k=args.eff_k,
        corrected=args.eff_corrected,
        weighted=args.eff_weighted,
        backend=args.eff_backend,
        input_metric=args.eff_input_metric,
        min_persistence=args.eff_min_persistence,
        max_holes=args.eff_max_holes,
        torch_chunk_size=args.torch_chunk_size,
        torch_device=args.torch_device,
        boundary_neighbor_k=args.eff_boundary_neighbor_k,
        boundary_radius_scale=args.eff_boundary_radius_scale,
        landmark_count=args.eff_landmarks,
        landmark_method=args.eff_landmark_method,
        use_witness_complex=args.eff_use_witness_complex,
        witness_k=args.eff_witness_k,
        witness_batch_size=args.eff_witness_batch,
    )

    path = Path(args.input)
    X = load_data(path, args.columns, args.delimiter)
    orig_d = X.shape[1]

    # Optional global subsample before any projection/landmarks
    subsample_indices = None
    if args.eff_subsample and args.eff_subsample > 0 and len(X) > args.eff_subsample:
        idx = rng.choice(len(X), size=args.eff_subsample, replace=False)
        X = X[idx]
        subsample_indices = idx
        logger.info(
            f"Subsampled input to {len(X)} points for ER-PH (requested eff_subsample={args.eff_subsample})"
        )

    if args.pca_components:
        n_comp = min(args.pca_components, X.shape[1])
        subset_size = min(args.pca_fit_samples, len(X))
        X_fit = (
            X
            if subset_size == len(X)
            else X[rng.choice(len(X), size=subset_size, replace=False)]
        )
        pca = PCA(
            n_components=n_comp,
            whiten=args.pca_whiten,
            svd_solver="randomized",
            random_state=args.seed,
        )
        pca.fit(X_fit)
        X = pca.transform(X)
        explained = getattr(pca, "explained_variance_ratio_", None)
        explained_cum = (
            float(np.sum(explained)) if explained is not None else float("nan")
        )
        logger.info(
            "Applied PCA: components={} (whiten={}) | fit_subset={} | original_d={} | explained_sum={:.4f}",
            n_comp,
            args.pca_whiten,
            subset_size,
            orig_d,
            explained_cum,
        )

    detector = EffectiveResistancePHDetector(**detector_kwargs)
    detector.fit(X)
    result = detector.detect_holes()
    metrics = compute_metrics(detector)

    # Base path for outputs (align everything with the input location by default)
    base_path = Path(args.output) if args.output else Path(args.input)

    # Optional diagnostic: save H1 diagram and cocycles (if present)
    if args.eff_save_diagram:
        diag_path = base_path.with_suffix(".ph_diag.npz")
        try:
            dgms = result.persistence_intervals
            np.savez_compressed(
                diag_path,
                h1_diagram=dgms if dgms is not None else np.empty((0, 2)),
                hole_centers=result.hole_centers,
                hole_sizes=result.hole_sizes,
            )
            logger.info(f"Saved PH diagnostic to {diag_path}")
        except Exception as e:  # pragma: no cover - best-effort
            logger.warning(f"Could not save PH diagnostic: {e}")

    logger.info("\n{}", format_metrics(metrics))

    centers = result.hole_centers
    sizes = (
        result.hole_sizes
        if len(result.hole_sizes)
        else np.zeros(len(centers), dtype=int)
    )
    sizes_subsample = (
        getattr(result, "hole_sizes_subsample", None)
        if hasattr(result, "hole_sizes_subsample")
        else None
    )
    if sizes_subsample is None or not len(sizes_subsample):
        sizes_subsample = np.zeros(len(centers), dtype=int)
    hole_subsample_indices = getattr(result, "hole_subsample_indices", [])
    nn_k = max(1, int(args.nn_neighbors))
    nn = NearestNeighbors(n_neighbors=min(nn_k, len(X))) if len(centers) else None
    if nn is not None:
        nn.fit(X)
        dists, idxs = nn.kneighbors(centers)
    else:
        dists = np.empty((0, 0))
        idxs = np.empty((0, 0), dtype=int)

    if len(centers):
        summary_lines = [
            (
                f"Hole {hid}: "
                f"landmarks={int(sizes[hid]) if len(sizes) > hid else 0} | "
                f"subsample={int(getattr(result, 'hole_sizes_subsample', sizes)[hid]) if len(sizes) > hid else 0} | "
                f"nn_idx={idxs[hid].tolist()}"
            )
            for hid, sz in enumerate(sizes)
        ]
        logger.info("Hole summary:\n{}", "\n".join(summary_lines))
    else:
        logger.info("Hole summary: no holes detected.")

    if centers.size == 0:
        coords_df = pd.DataFrame()
    else:
        coords_df = pd.DataFrame(
            centers, columns=[f"x{i+1}" for i in range(centers.shape[1])]
        )

    # Detailed CSV per hole (empty if none)
    detail_rows = []
    for hid, center in enumerate(centers):
        row = {
            "hole_id": hid,
            "size_landmarks": int(sizes[hid]) if len(sizes) > hid else 0,
            "size_subsample": (
                int(sizes_subsample[hid]) if len(sizes_subsample) > hid else 0
            ),
            "nn_indices": json.dumps(idxs[hid].tolist()) if len(idxs) > hid else "[]",
            "nn_distances": (
                json.dumps(dists[hid].tolist()) if len(dists) > hid else "[]"
            ),
        }
        row.update({f"c{i+1}": float(c) for i, c in enumerate(center)})
        detail_rows.append(row)
    detail_df = pd.DataFrame(detail_rows)

    # Save outputs (default alongside input if --output not provided)
    output_path = base_path.with_suffix(".hole_centers.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords_df.to_csv(output_path, index=False)
    metrics_path = base_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    sample_path = base_path.with_suffix(".samples.npz")
    grouped = {
        f"hole_{i}_samples": samples
        for i, samples in enumerate(result.hole_sample_groups)
    }
    grouped["boundary_points"] = np.array(
        result.component_boundary_points, dtype=object
    )
    grouped["boundary_indices"] = np.array(
        getattr(result, "component_boundary_indices", []), dtype=object
    )
    grouped["boundary_distances"] = np.array(
        getattr(result, "component_boundary_distances", []), dtype=object
    )
    grouped["hole_sizes_subsample"] = sizes_subsample
    grouped["hole_subsample_indices"] = np.array(hole_subsample_indices, dtype=object)
    if subsample_indices is not None:
        grouped["subsample_indices"] = subsample_indices
    np.savez_compressed(
        sample_path,
        centers=result.hole_centers,
        sizes=result.hole_sizes,
        hole_samples=result.hole_samples,
        cluster_labels=result.cluster_labels,
        maxima=result.maxima,
        **grouped,
    )
    logger.info("Saved hole centers to {}", output_path)
    logger.info("Saved metrics to {}", metrics_path)
    logger.info("Saved sampling details to {}", sample_path)

    summary_csv_path = base_path.with_suffix(".holes.csv")
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved hole summary CSV to {}", summary_csv_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    if args.command == "synthetic":
        synthetic_experiments(args)
    elif args.command == "apply":
        apply_detector(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

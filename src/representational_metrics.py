from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Visualization imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.analysis_plus import l2_normalize


def nature_style():
    """Matplotlib rc context for Nature-style figures."""
    return plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


@dataclass
class ModelInfo:
    """Information about a detected model and its embeddings."""

    name: str
    model_path: Optional[str]
    embedding_path: str
    config_path: Optional[str] = None
    embedding_dim: Optional[int] = None
    num_samples: Optional[int] = None
    datasets: Optional[List[str]] = None


class RepresentationalMetrics:
    """Compute representational similarity metrics between embedding spaces."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute Linear Centered Kernel Alignment (CKA) between two representations.
        CKA works with different embedding dimensions by comparing Gram matrices.

        Args:
            X: First representation matrix (n_samples, n_features_x)
            Y: Second representation matrix (n_samples, n_features_y)

        Returns:
            CKA score between 0 and 1
        """

        def center_gram_matrix(K):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ K @ H

        # Normalize representations
        X = l2_normalize(X)
        Y = l2_normalize(Y)

        # Compute gram matrices (n_samples x n_samples)
        K_X = X @ X.T
        K_Y = Y @ Y.T

        # Center the matrices
        K_X = center_gram_matrix(K_X)
        K_Y = center_gram_matrix(K_Y)

        # Compute CKA
        numerator = np.trace(K_X @ K_Y)
        denominator = np.sqrt(np.trace(K_X @ K_X) * np.trace(K_Y @ K_Y))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def compute_nearest_neighbors(self, feats: np.ndarray, topk: int = 1) -> np.ndarray:
        """
        Compute the nearest neighbors of feats
        Args:
            feats: a numpy array of shape N x D
            topk: the number of nearest neighbors to return
        Returns:
            knn: a numpy array of shape N x topk
        """
        # Normalize features
        feats_norm = l2_normalize(feats)

        # Compute similarity matrix
        sim_matrix = feats_norm @ feats_norm.T

        # Fill diagonal with very negative values to exclude self
        np.fill_diagonal(sim_matrix, -1e8)

        # Get top-k indices (highest similarity = nearest neighbors)
        knn = np.argsort(sim_matrix, axis=1)[:, -topk:][:, ::-1]  # Sort descending

        return knn

    def mutual_knn(
        self,
        feats_A: np.ndarray,
        feats_B: np.ndarray,
        topk: int = 10,
    ) -> float:
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A numpy array of shape N x feat_dim_A
            feats_B: A numpy array of shape N x feat_dim_B (can have different feat_dim)
            topk: Number of nearest neighbors

        Returns:
            A float representing the mutual KNN accuracy
        """
        # No need to project - we just compare which objects are nearest neighbors
        knn_A = self.compute_nearest_neighbors(feats_A, topk)
        knn_B = self.compute_nearest_neighbors(feats_B, topk)

        n = knn_A.shape[0]

        # Create binary masks for knn_A and knn_B
        lvm_mask = np.zeros((n, n))
        llm_mask = np.zeros((n, n))

        # Set masks for nearest neighbors
        for i in range(n):
            lvm_mask[i, knn_A[i]] = 1.0
            llm_mask[i, knn_B[i]] = 1.0

        # Compute accuracy: intersection of neighbors divided by topk
        acc = (lvm_mask * llm_mask).sum(axis=1) / topk

        return acc.mean()

    def bootstrap_metric(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        metric_fn: Callable,
        n_bootstrap: int = 100,
        sample_size: int = 1000,
        **metric_kwargs,
    ) -> Dict[str, float]:
        """
        Compute bootstrapped statistics for a representational metric.

        Args:
            X: First representation matrix
            Y: Second representation matrix
            metric_fn: Function to compute the metric
            n_bootstrap: Number of bootstrap samples
            sample_size: Size of each bootstrap sample
            **metric_kwargs: Additional arguments for metric function

        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        n_samples = min(X.shape[0], Y.shape[0])
        sample_size = min(sample_size, n_samples)

        scores = []
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
            # Sample indices
            indices = self.rng.choice(n_samples, size=sample_size, replace=False)

            # Compute metric on sample
            X_sample = X[indices]
            Y_sample = Y[indices]
            score = metric_fn(X_sample, Y_sample, **metric_kwargs)
            scores.append(score)

        scores = np.array(scores)
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_lower": float(np.percentile(scores, 2.5)),
            "ci_upper": float(np.percentile(scores, 97.5)),
            "median": float(np.median(scores)),
            "n_bootstrap": n_bootstrap,
            "sample_size": sample_size,
        }


class RepresentationalVisualization:
    """Create visualizations for representational metrics analysis."""

    def __init__(self, output_dir: str = "assets/representational_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _get_model_ranking_order(self):
        """Define the ranking order for models based on performance."""
        return [
            "I-T: CLIP Random fine-tuned",
            "I-T: CLIP LAION",
            "I-T: CLIP LAION fine-tuned",
            "I-T: MONET",
            "I-T: MONET fine-tuned",
            "SSL: MAE",
            "SSL: DINO",
            "SSL: iBOT",
            "I-T: Ensemble (d = 1,792)",
            "I-T: Ensemble (d = 512)",
            "I-T: Ensemble (d = 256)",
            "I-T: Ensemble (d = 128)",
            "Ensemble (d = 3,840)",
            "Ensemble (d = 1,024)",
            "Ensemble (d = 512)",
            "Ensemble (d = 256)",
        ]

    def plot_similarity_matrices(
        self,
        results_df: pd.DataFrame,
        save_plots: bool = True,
    ):
        """Create separate heatmaps for CKA and mutual k-NN similarity matrices with proper ranking order."""

        # Get unique model names and apply ranking order
        all_models = set(
            results_df["model_a"].tolist() + results_df["model_b"].tolist()
        )
        display_names_map = {
            name: self._shorten_model_name(name) for name in all_models
        }

        # Get ranking order and filter to only include models we have
        ranking_order = self._get_model_ranking_order()
        models = []
        for ranked_name in ranking_order:
            for model_name, display_name in display_names_map.items():
                if display_name == ranked_name:
                    models.append(model_name)
                    break

        # Add any remaining models not in ranking
        for model_name in all_models:
            if model_name not in models:
                models.append(model_name)

        n_models = len(models)

        # Create similarity matrices
        cka_matrix = np.eye(n_models)
        knn_matrix = np.eye(n_models)

        # Fill matrices
        for _, row in results_df.iterrows():
            i = models.index(row["model_a"])
            j = models.index(row["model_b"])

            cka_matrix[i, j] = row["cka_mean"]
            cka_matrix[j, i] = row["cka_mean"]  # Symmetric

            knn_matrix[i, j] = row["mutual_knn_mean"]
            knn_matrix[j, i] = row["mutual_knn_mean"]

        # Get display names in correct order
        display_names = [self._shorten_model_name(name) for name in models]

        # Create separate plots for each metric
        with nature_style():
            # CKA heatmap
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(display_names, rotation=45, ha="right")
            ax.set_yticklabels(display_names)
            ax.grid(False)

            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    ax.text(
                        j,
                        i,
                        f"{cka_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if cka_matrix[i, j] < 0.5 else "black",
                        fontsize=8,
                    )

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("CKA Score", rotation=270, labelpad=20)

            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    self.output_dir / "cka_similarity_matrix.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
            plt.close()

            # Mutual k-NN heatmap
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(knn_matrix, cmap="plasma", vmin=0, vmax=1)
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(display_names, rotation=45, ha="right")
            ax.set_yticklabels(display_names)
            ax.grid(False)

            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    ax.text(
                        j,
                        i,
                        f"{knn_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if knn_matrix[i, j] < 0.5 else "black",
                        fontsize=8,
                    )

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Mutual k-NN Score", rotation=270, labelpad=20)

            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    self.output_dir / "mutual_knn_similarity_matrix.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
            plt.close()

    def plot_bootstrap_convergence(
        self,
        results_df: pd.DataFrame,
        save_plots: bool = True,
    ):
        """Plot bootstrap confidence intervals to show statistical robustness with separate plots."""
        # Select top 6 most similar pairs for visualization
        top_pairs = results_df.nlargest(6, "cka_mean")

        with nature_style():
            # CKA confidence intervals (separate plot)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            y_pos = np.arange(len(top_pairs))
            pair_labels = [
                f"{self._shorten_model_name(row['model_a'])} vs\n{self._shorten_model_name(row['model_b'])}"
                for _, row in top_pairs.iterrows()
            ]

            ax.barh(
                y_pos,
                top_pairs["cka_mean"],
                xerr=[
                    top_pairs["cka_mean"] - top_pairs["cka_ci_lower"],
                    top_pairs["cka_ci_upper"] - top_pairs["cka_mean"],
                ],
                capsize=5,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=10)
            ax.set_xlabel("CKA Score")
            ax.set_title(
                "CKA Scores with 95% Confidence Intervals\n(Top 6 Most Similar Pairs)",
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    self.output_dir / "cka_bootstrap_confidence.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
            plt.close()

            # Mutual k-NN confidence intervals (separate plot)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Sort by mutual k-NN for this plot
            top_pairs_knn = results_df.nlargest(6, "mutual_knn_mean")
            y_pos = np.arange(len(top_pairs_knn))
            pair_labels_knn = [
                f"{self._shorten_model_name(row['model_a'])} vs\n{self._shorten_model_name(row['model_b'])}"
                for _, row in top_pairs_knn.iterrows()
            ]

            ax.barh(
                y_pos,
                top_pairs_knn["mutual_knn_mean"],
                xerr=[
                    top_pairs_knn["mutual_knn_mean"]
                    - top_pairs_knn["mutual_knn_ci_lower"],
                    top_pairs_knn["mutual_knn_ci_upper"]
                    - top_pairs_knn["mutual_knn_mean"],
                ],
                capsize=5,
                alpha=0.7,
                color="darkgreen",
                edgecolor="black",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels_knn, fontsize=10)
            ax.set_xlabel("Mutual k-NN Score")
            ax.set_title(
                "Mutual k-NN Scores with 95% Confidence Intervals\n(Top 6 Most Similar Pairs)",
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_plots:
                plt.savefig(
                    self.output_dir / "mutual_knn_bootstrap_confidence.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()
            plt.close()

    def _shorten_model_name(self, name: str) -> str:
        """Shorten long model names for better display."""
        if "combined" in name:
            dim = None
            if "svd" in name:
                dim = name.split("_")[-1].replace("svd", "")

            if "and_3_more" in name:
                if dim is None:
                    dim = "3840"
                name = f"Ensemble (d = {int(dim):,})"
            else:
                if dim is None:
                    dim = "1792"
                name = f"I-T: Ensemble (d = {int(dim):,})"

        if "random" in name:
            name = "I-T: CLIP Random fine-tuned"

        if "openai_clip-vit-base-patch32" in name:
            if "openai_clip-vit-base-patch32-" in name:
                return "I-T: CLIP LAION fine-tuned"
            else:
                return "I-T: CLIP LAION"

        if "suinleelab_monet" in name:
            if "suinleelab_monet-" in name:
                return "I-T: MONET fine-tuned"
            else:
                return "I-T: MONET"

        # Handle DINO variants
        if "ssl_dino_qderma" in name:
            return "SSL: DINO"
        elif "ssl_ibot_qderma" in name:
            return "SSL: iBOT"
        elif "ssl_mae_qderma" in name:
            return "SSL: MAE"

        # Truncate very long names
        if len(name) > 30:
            name = name[:27] + "..."

        return name

    def generate_all_visualizations(self, results_df: pd.DataFrame, models: List):
        """Generate all visualizations and summary report."""
        self.plot_similarity_matrices(results_df)
        self.plot_bootstrap_convergence(results_df)
        print(f"All visualizations saved to {self.output_dir}/")


class ModelDetector:
    """Detect and analyze models in the assets directory."""

    def __init__(self, assets_dir: str):
        self.assets_dir = Path(assets_dir)
        self.models = []

    def detect_models(self) -> List[ModelInfo]:
        """
        Detect all models with embeddings in the assets directory.

        Returns:
            List of ModelInfo objects
        """
        self.models = []

        for item in self.assets_dir.iterdir():
            if not item.is_dir():
                continue

            model_info = self._analyze_model_directory(item)
            if model_info:
                self.models.append(model_info)

        return self.models

    def _analyze_model_directory(self, model_dir: Path) -> Optional[ModelInfo]:
        """Analyze a single model directory."""

        # Look for embeddings
        embedding_path = None

        # Check for embeddings in subdirectories
        embeddings_dir = model_dir / "embeddings"
        if embeddings_dir.exists():
            # Look for .npz files
            npz_files = list(embeddings_dir.glob("*.npz"))
            if npz_files:
                embedding_path = str(npz_files[0])  # Use first .npz file

        # Also check for .npz files directly in model directory
        if not embedding_path:
            npz_files = list(model_dir.glob("*.npz"))
            if npz_files:
                embedding_path = str(npz_files[0])

        if not embedding_path:
            return None

        # Look for model files
        model_path = None
        model_files = (
            list(model_dir.glob("**/model.safetensors"))
            + list(model_dir.glob("**/pytorch_model.bin"))
            + list(model_dir.glob("**/model.bin"))
        )
        if model_files:
            model_path = str(model_files[0])

        # Look for config
        config_path = None
        config_files = list(model_dir.glob("**/config.json"))
        if config_files:
            config_path = str(config_files[0])

        # Try to get embedding info
        embedding_dim, num_samples, datasets = self._get_embedding_info(embedding_path)

        return ModelInfo(
            name=model_dir.name,
            model_path=model_path,
            embedding_path=embedding_path,
            config_path=config_path,
            embedding_dim=embedding_dim,
            num_samples=num_samples,
            datasets=datasets,
        )

    def _get_embedding_info(
        self, embedding_path: str
    ) -> Tuple[Optional[int], Optional[int], Optional[List[str]]]:
        """Get information about embeddings file."""
        try:
            # Load embeddings to get dimensions
            data = np.load(embedding_path)

            # Different possible keys in the npz file
            embedding_keys = [
                "embeddings",
                "image_embeddings",
                "features",
                "representations",
            ]
            embeddings = None

            for key in embedding_keys:
                if key in data:
                    embeddings = data[key]
                    break

            if embeddings is None and len(data.files) > 0:
                # Use first available array
                embeddings = data[data.files[0]]

            if embeddings is not None:
                embedding_dim = (
                    embeddings.shape[1]
                    if len(embeddings.shape) > 1
                    else embeddings.shape[0]
                )
                num_samples = embeddings.shape[0] if len(embeddings.shape) > 1 else 1

                # Try to get dataset info from corresponding dataframe
                df_path = Path(embedding_path).parent / "dataframe.csv"
                datasets = None
                if df_path.exists():
                    try:
                        df = pd.read_csv(df_path)
                        if "dataset" in df.columns:
                            datasets = df["dataset"].unique().tolist()
                    except Exception:
                        pass

                return embedding_dim, num_samples, datasets

        except Exception as e:
            print(f"Warning: Could not load embeddings from {embedding_path}: {e}")

        return None, None, None


def load_embeddings(
    model_info: ModelInfo,
    max_samples: Optional[int] = None,
    sample_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load embeddings from a model."""
    data = np.load(model_info.embedding_path)

    # Find embeddings in the npz file
    embedding_keys = ["embeddings", "image_embeddings", "features", "representations"]
    embeddings = None

    for key in embedding_keys:
        if key in data:
            embeddings = data[key]
            break

    if embeddings is None and len(data.files) > 0:
        embeddings = data[data.files[0]]

    if embeddings is None:
        raise ValueError(f"No embeddings found in {model_info.embedding_path}")

    # Sample if too large - use provided indices or generate once
    if max_samples and embeddings.shape[0] > max_samples:
        if sample_indices is not None:
            embeddings = embeddings[sample_indices]
        else:
            indices = np.random.choice(embeddings.shape[0], max_samples, replace=False)
            embeddings = embeddings[indices]

    return embeddings.astype(np.float32)


def main(
    generate_visualizations: bool = True,
    max_samples_per_model: int = 10_000,
    bootstrap_samples: int = 50,
    output_dir: str = "assets/representational_analysis",
):
    """Main function to run representational analysis."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Detecting models in assets directory...")
    detector = ModelDetector("assets")
    models = detector.detect_models()

    print(f"Found {len(models)} models with embeddings:")
    for model in models:
        print(
            f"  - {model.name}: {model.num_samples} samples, dim={model.embedding_dim}"
        )

    if len(models) < 2:
        print("Need at least 2 models to compute similarity metrics")
        return

    # Initialize metrics computer
    metrics = RepresentationalMetrics()

    # Load all embeddings with same sample indices
    print("\nLoading embeddings...")
    embeddings_data = {}

    # Generate sample indices once for all models to ensure same images
    sample_indices = None
    max_available_samples = min(
        [model.num_samples for model in models if model.num_samples]
    )
    if max_samples_per_model and max_available_samples > max_samples_per_model:
        np.random.seed(42)  # Fixed seed for reproducibility
        sample_indices = np.random.choice(
            max_available_samples, max_samples_per_model, replace=False
        )
        print(f"  Using same {len(sample_indices)} sample indices for all models")

    for model in models:
        try:
            emb = load_embeddings(model, max_samples_per_model, sample_indices)
            embeddings_data[model.name] = emb
            print(f"  {model.name}: {emb.shape}")
        except Exception as e:
            print(f"  Error loading {model.name}: {e}")

    # Compute pairwise metrics
    print("\nComputing representational metrics...")
    results = []

    model_names = list(embeddings_data.keys())
    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            if i >= j:  # Skip duplicates and self-comparison
                continue

            print(f"\n  {model_a} vs {model_b}")

            X = embeddings_data[model_a]
            Y = embeddings_data[model_b]

            # Ensure same number of samples
            min_samples = min(X.shape[0], Y.shape[0])
            # X = X[:min_samples]
            # Y = Y[:min_samples]

            # Compute metrics with bootstrapping
            try:
                # Linear CKA
                print("    Computing CKA...")
                cka_stats = metrics.bootstrap_metric(
                    X,
                    Y,
                    metrics.linear_cka,
                    n_bootstrap=bootstrap_samples,
                    sample_size=min(1000, min_samples),
                )

                # Mutual k-NN overlap
                print("    Computing mutual k-NN...")
                knn_stats = metrics.bootstrap_metric(
                    X,
                    Y,
                    metrics.mutual_knn,
                    n_bootstrap=bootstrap_samples,
                    sample_size=min(1000, min_samples),
                    topk=100,
                )

                result = {
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_samples": min_samples,
                    "cka_mean": cka_stats["mean"],
                    "cka_std": cka_stats["std"],
                    "cka_ci_lower": cka_stats["ci_lower"],
                    "cka_ci_upper": cka_stats["ci_upper"],
                    "mutual_knn_mean": knn_stats["mean"],
                    "mutual_knn_std": knn_stats["std"],
                    "mutual_knn_ci_lower": knn_stats["ci_lower"],
                    "mutual_knn_ci_upper": knn_stats["ci_upper"],
                }

                results.append(result)

                print(f"    CKA: {cka_stats['mean']:.3f} ± {cka_stats['std']:.3f}")
                print(
                    f"    Mutual k-NN: {knn_stats['mean']:.3f} ± {knn_stats['std']:.3f}"
                )

            except Exception as e:
                print(f"    Error computing metrics: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_path = Path(output_dir) / "representational_metrics_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Print summary
        print(f"\nSummary of {len(results)} model pairs:")
        print(
            results_df[["model_a", "model_b", "cka_mean", "mutual_knn_mean"]].to_string(
                index=False
            )
        )

        # Generate visualizations if requested
        if generate_visualizations:
            print("\nGenerating visualizations...")
            visualizer = RepresentationalVisualization(output_dir)
            visualizer.generate_all_visualizations(results_df, models)

    else:
        print("\nNo results to save")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute representational similarity metrics between models"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Maximum samples per model (default: 10,000)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=50,
        help="Number of bootstrap samples (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/representational_analysis",
        help="Output directory for visualizations (default: representational_analysis)",
    )

    args = parser.parse_args()

    # Update main function to accept these parameters
    main(
        generate_visualizations=not args.no_visualizations,
        max_samples_per_model=args.max_samples,
        bootstrap_samples=args.bootstrap_samples,
        output_dir=args.output_dir,
    )

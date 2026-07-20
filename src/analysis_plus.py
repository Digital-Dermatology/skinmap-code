"""
analysis_plus.py (minimal version)

Contains only:
- Yearly Novelty analysis
- Domain Shift (Fréchet distance) analysis
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    import faiss  # type: ignore
except ImportError as e:
    raise ImportError(
        "FAISS is required by analysis_plus. Install with: pip install faiss-cpu"
    ) from e

from scipy.linalg import sqrtm

# -------------------------- Core utilities --------------------------


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def build_faiss_index(X: np.ndarray, metric: str = "cosine") -> faiss.Index:
    d = X.shape[1]
    if metric == "cosine":
        idx = faiss.IndexFlatIP(d)
    elif metric == "l2":
        idx = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")
    idx.add(X.astype(np.float32))
    return idx


# -------------------------- Domain shift (Fréchet) --------------------------


def _mean_and_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = X.mean(axis=0)
    Xc = X - m
    C = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    return m, C


def frechet_distance_gaussians(m1, C1, m2, C2) -> float:
    diff = m1 - m2
    try:
        sqrt_prod = sqrtm(C1 @ C2)
        if np.iscomplexobj(sqrt_prod):
            sqrt_prod = sqrt_prod.real
    except Exception:
        sqrt_prod = np.diag(
            np.sqrt(np.clip(np.diag(C1), 0, None))
            * np.sqrt(np.clip(np.diag(C2), 0, None))
        )
    fd = diff @ diff + np.trace(C1 + C2 - 2.0 * sqrt_prod)
    return float(np.real(fd))


def dataset_domain_shift(X: np.ndarray, datasets: Iterable) -> pd.DataFrame:
    df = pd.DataFrame({"dataset": list(datasets)})
    Xn = l2_normalize(X)
    groups = df.groupby("dataset").indices
    stats = {k: _mean_and_cov(Xn[list(idxs)]) for k, idxs in groups.items()}
    rows = []
    keys = list(stats.keys())
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if j <= i:
                continue
            m1, C1 = stats[a]
            m2, C2 = stats[b]
            fd = frechet_distance_gaussians(m1, C1, m2, C2)
            rows.append({"dataset_a": a, "dataset_b": b, "frechet": fd})
    return pd.DataFrame(rows)


# -------------------------- Yearly novelty --------------------------


def yearly_novelty(
    X: np.ndarray,
    years: Iterable[int],
    k: int = 10,
    prebuilt_index=None,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int | None = None,
    bootstrap_sample_size: int | None = None,
    max_bootstrap_queries: int | None = None,
) -> pd.DataFrame:
    Xn = l2_normalize(X)
    years = np.array(list(years))
    uniq = np.sort(np.unique(years))
    rows = []
    rng = np.random.default_rng(random_state)

    for y in tqdm(uniq, desc="Yearly novelty", leave=False):
        cur_idx = np.where(years == y)[0]
        prev_idx = np.where(years < y)[0]
        cur_count = len(cur_idx)
        if len(prev_idx) == 0 or len(cur_idx) == 0:
            rows.append(
                {
                    "year": int(y),
                    "raw_novelty": np.nan,
                    "norm_novelty": np.nan,
                    "baseline_novelty": np.nan,
                    "baseline_ci_low": np.nan,
                    "baseline_ci_high": np.nan,
                    "novelty_vs_baseline": np.nan,
                    "n": int(cur_count),
                }
            )
            continue
        idx = build_faiss_index(Xn[prev_idx], metric="cosine")
        sims, _ = idx.search(Xn[cur_idx].astype(np.float32), min(k, len(prev_idx)))
        # Convert cosine similarity to a per-sample mean distance
        per_sample = (1.0 - sims).mean(axis=1)
        raw = float(per_sample.mean())

        # Null baseline: expected novelty if we drew the same number of samples
        # from the pool of previous years.
        baseline_vals = []
        if n_bootstrap > 0 and len(prev_idx) > 0:
            sample_size = bootstrap_sample_size or cur_count
            sample_size = max(1, int(sample_size))
            # If the query cap would zero-out reps, shrink the draw size so that
            # at least one baseline sample is computed.
            if (
                max_bootstrap_queries is not None
                and sample_size > max_bootstrap_queries
            ):
                sample_size = max(1, int(max_bootstrap_queries))
            # Optional cap to prevent very large total searches when prev_idx or
            # bootstrap reps are huge.
            reps = n_bootstrap
            if max_bootstrap_queries is not None:
                allowed = max_bootstrap_queries // max(1, sample_size)
                allowed = max(1, allowed)
                reps = min(reps, allowed)
            for _ in tqdm(range(reps), desc=f"Baseline y={int(y)}", leave=False):
                boot_idx = rng.choice(prev_idx, size=sample_size, replace=True)
                queries = Xn[boot_idx].astype(np.float32)
                # Drop the self-match by asking for one extra neighbor.
                k_eff = min(k + 1, len(prev_idx))
                boot_sims, _ = idx.search(queries, k_eff)
                boot_sims = boot_sims[:, 1:] if boot_sims.shape[1] > 1 else boot_sims
                if boot_sims.shape[1] == 0:
                    continue
                boot_sims = boot_sims[:, :k]
                boot_per_sample = (1.0 - boot_sims).mean(axis=1)
                baseline_vals.append(float(boot_per_sample.mean()))

        baseline_mean = float(np.nanmean(baseline_vals)) if baseline_vals else np.nan
        ci_low = (
            float(np.nanquantile(baseline_vals, alpha / 2.0))
            if baseline_vals
            else np.nan
        )
        ci_high = (
            float(np.nanquantile(baseline_vals, 1.0 - alpha / 2.0))
            if baseline_vals
            else np.nan
        )
        novelty_vs_baseline = (
            float(raw / baseline_mean)
            if np.isfinite(baseline_mean) and baseline_mean > 0
            else np.nan
        )

        rows.append(
            {
                "year": int(y),
                "raw_novelty": raw,
                # Keep the legacy column name but without the previous up-weighting.
                "norm_novelty": raw,
                "baseline_novelty": baseline_mean,
                "baseline_ci_low": ci_low,
                "baseline_ci_high": ci_high,
                "novelty_vs_baseline": novelty_vs_baseline,
                "n": int(cur_count),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "l2_normalize",
    "build_faiss_index",
    "dataset_domain_shift",
    "yearly_novelty",
]

"""Effective-resistance-based persistent homology hole detection.

This module adapts the spectral/effective-resistance distance proposed in
the eff-ph paper to the SkinMap pipeline. It supports a NumPy path and a
torch path (GPU-capable for the kNN construction). Persistent homology is
computed with ripser on the resulting distance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch  # torch is a dependency in requirements
from loguru import logger
from ripser import ripser
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _knn_numpy(
    X: np.ndarray, k: int, metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dists, indices = nn.kneighbors(X)
    return indices[:, 1:], dists[:, 1:]


def _knn_torch(
    X: "torch.Tensor", k: int, chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Blockwise kNN on GPU to avoid full n×n distance allocation."""
    X = X.float()
    n = X.shape[0]
    q_chunk = max(1, chunk_size or 2048)
    ref_chunk = q_chunk
    all_idx: List[torch.Tensor] = []
    all_dist: List[torch.Tensor] = []
    q_iter = range(0, n, q_chunk)
    q_iter = tqdm(q_iter, desc="kNN (torch)", leave=False) if n > q_chunk else q_iter
    for q_start in q_iter:
        q_end = min(n, q_start + q_chunk)
        q = X[q_start:q_end]
        best_dist = torch.full((len(q), k), float("inf"), device=X.device)
        best_idx = torch.full((len(q), k), -1, device=X.device, dtype=torch.long)
        for r_start in range(0, n, ref_chunk):
            r_end = min(n, r_start + ref_chunk)
            ref = X[r_start:r_end]
            dist = torch.cdist(q, ref)
            # mask self distances when query/ref overlap
            if q_start == r_start:
                rows = torch.arange(len(q), device=X.device)
                dist[rows, rows] = float("inf")
            cand_dist, cand_idx_local = torch.topk(dist, k, largest=False, dim=1)
            cand_idx = cand_idx_local + r_start
            merged_dist = torch.cat([best_dist, cand_dist], dim=1)
            merged_idx = torch.cat([best_idx, cand_idx], dim=1)
            best_dist, top_idx = torch.topk(merged_dist, k, largest=False, dim=1)
            best_idx = merged_idx.gather(1, top_idx)
        all_idx.append(best_idx.cpu())
        all_dist.append(best_dist.cpu())
    indices = torch.cat(all_idx, dim=0).numpy()
    dists = torch.cat(all_dist, dim=0).numpy()
    return indices, dists


def _symmetrize_knn(
    indices: np.ndarray, dists: np.ndarray, weighted: bool
) -> sp.coo_matrix:
    n, k = indices.shape
    rows = np.repeat(np.arange(n), k)
    cols = indices.flatten()
    data = dists.flatten() if weighted else np.ones_like(rows, dtype=float)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.transpose()).tocoo()
    return adj


def _effective_resistance_from_adj(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    degs = adj.sum(axis=1)
    L = np.diag(degs) - adj
    J = np.ones((n, n), dtype=L.dtype) / n
    # Von Luxburg style regularization to handle singular Laplacian.
    Lpinv = np.linalg.inv(L + J) - J
    diag = np.diag(Lpinv)
    eff = diag[:, None] + diag[None, :] - 2 * Lpinv
    np.fill_diagonal(eff, 0.0)
    return eff


def _correct_effective_resistance(d: np.ndarray, adj: np.ndarray) -> np.ndarray:
    degs = adj.sum(axis=1)
    deg_dist = 1.0 / degs[:, None] + 1.0 / degs[None, :]
    np.fill_diagonal(deg_dist, 0.0)
    # The +2*adj/(degs*degs.T) term follows the von Luxburg correction.
    correction = deg_dist - 2.0 * adj / (degs[:, None] * degs[None, :])
    return d - correction


def effective_resistance_distance(
    X: ArrayLike,
    k: int = 15,
    *,
    corrected: bool = True,
    weighted: bool = False,
    backend: str = "numpy",
    input_metric: str = "euclidean",
    disconnect: bool = True,
    torch_chunk_size: Optional[int] = None,
    torch_device: Optional[str] = None,
) -> np.ndarray:
    """Compute effective resistance distances on a symmetric kNN graph.

    Parameters
    ----------
    X : array-like (n, d)
        Input data. Accepts numpy arrays or torch tensors.
    k : int
        Number of neighbors (excluding self).
    corrected : bool
        Apply von Luxburg degree correction.
    weighted : bool
        Use edge weights from kNN distances (True) or unweighted graph (False).
    backend : {"numpy", "torch"}
        Backend for kNN construction. Effective resistance itself is computed
        on CPU; the torch path accelerates the neighbor search (and keeps data
        on GPU until the Laplacian step).
    input_metric : str
        Distance metric for kNN search.
    disconnect : bool
        Compute components separately and set inter-component distances to a
        large value.
    torch_chunk_size : Optional[int]
        Chunk size for torch cdist to manage memory.
    torch_device : Optional[str]
        Device to move torch tensors to (e.g., "cuda"). Defaults to X.device or CPU.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if backend not in {"numpy", "torch"}:
        raise ValueError("backend must be 'numpy' or 'torch'.")

    n_points = X.shape[0] if isinstance(X, np.ndarray) else int(X.shape[0])
    if k >= n_points:
        raise ValueError("k must be smaller than the number of samples.")

    use_torch_knn = (
        backend == "torch"
        and torch is not None
        and (
            (torch_device and torch_device != "cpu")
            or (isinstance(X, torch.Tensor) and X.device.type != "cpu")
        )
    )

    if use_torch_knn:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.asarray(X), device=torch_device or "cuda")
        elif torch_device and X.device.type != torch_device:
            X = X.to(torch_device)
        indices, dists = _knn_torch(X, k=k, chunk_size=torch_chunk_size)
    else:
        X_np = (
            X.detach().cpu().numpy()
            if torch is not None and isinstance(X, torch.Tensor)
            else np.asarray(X)
        )
        indices, dists = _knn_numpy(X_np, k=k, metric=input_metric)

    adj = _symmetrize_knn(indices, dists, weighted=weighted)

    if disconnect:
        n_components, labels = csgraph.connected_components(adj)
    else:
        n_components, labels = 1, np.zeros(adj.shape[0], dtype=int)

    eff = np.full(adj.shape, np.inf, dtype=float)
    for comp in range(n_components):
        mask = labels == comp
        if mask.sum() == 0:
            continue
        sub_adj = adj.tocsr()[mask][:, mask].toarray()
        eff_block = _effective_resistance_from_adj(sub_adj)
        eff[np.ix_(mask, mask)] = eff_block

    finite_mask = np.isfinite(eff)
    if not finite_mask.any():
        raise RuntimeError("Effective resistance matrix contains no finite entries.")
    max_finite = eff[finite_mask].max()
    eff[~finite_mask] = max_finite * 2.0

    if corrected:
        eff = _correct_effective_resistance(eff, adj.toarray())

    return eff


@dataclass
class EffectiveResistancePHResult:
    persistence_intervals: np.ndarray
    hole_centers: np.ndarray
    hole_sizes: np.ndarray
    hole_sizes_subsample: np.ndarray
    hole_samples: np.ndarray
    hole_sample_groups: List[np.ndarray]
    hole_subsample_indices: List[np.ndarray]
    filtered_holes: np.ndarray
    cluster_labels: np.ndarray
    maxima: np.ndarray
    component_boundary_points: List[np.ndarray]
    component_boundary_indices: List[np.ndarray]
    component_boundary_distances: List[np.ndarray]
    component_volumes: np.ndarray


class EffectiveResistancePHDetector:
    """Persistent-homology-based hole detector using effective resistance distances."""

    def __init__(
        self,
        k: int = 30,
        corrected: bool = True,
        weighted: bool = False,
        backend: str = "numpy",
        input_metric: str = "euclidean",
        maxdim: int = 1,
        min_persistence: float = 0.0,
        max_holes: Optional[int] = None,
        torch_chunk_size: Optional[int] = None,
        torch_device: Optional[str] = None,
        boundary_neighbor_k: int = 25,
        boundary_radius_scale: float = 1.5,
        landmark_count: Optional[int] = None,
        landmark_method: str = "random",
        use_witness_complex: bool = False,
        witness_k: int = 5,
        witness_batch_size: int = 5000,
        sparse_knn_k: Optional[int] = None,
    ) -> None:
        self.k = k
        self.corrected = corrected
        self.weighted = weighted
        self.backend = backend
        self.input_metric = input_metric
        self.maxdim = maxdim
        self.min_persistence = min_persistence
        self.max_holes = max_holes
        self.torch_chunk_size = torch_chunk_size
        self.torch_device = torch_device
        self.boundary_neighbor_k = max(1, int(boundary_neighbor_k))
        self.boundary_radius_scale = float(boundary_radius_scale)
        self.landmark_count = landmark_count
        self.landmark_method = landmark_method
        self.use_witness_complex = bool(use_witness_complex)
        self.witness_k = max(1, int(witness_k))
        self.witness_batch_size = max(1, int(witness_batch_size))
        self.sparse_knn_k = sparse_knn_k
        self.X: Optional[np.ndarray] = None
        self._distance: Optional[np.ndarray] = None
        self._landmark_assignments: Optional[np.ndarray] = None
        self._subsample_indices: Optional[np.ndarray] = None
        self._distance_sparse: Optional[sp.csr_matrix] = None

    def fit(self, X: ArrayLike) -> "EffectiveResistancePHDetector":
        X_np = (
            X.detach().cpu().numpy()
            if torch is not None and isinstance(X, torch.Tensor)
            else np.asarray(X)
        )
        self.X = np.asarray(X_np, dtype=np.float32)
        self._subsample_indices = np.arange(len(self.X))
        return self

    def _compute_distance(self) -> Union[np.ndarray, sp.csr_matrix]:
        if self.X is None:
            raise RuntimeError("Call fit(X) before detect_holes().")
        X_work = self.X
        dist: Optional[np.ndarray] = None
        if self.landmark_count is not None and 0 < self.landmark_count < len(X_work):
            logger.info(
                f"[ER-PH] Selecting {self.landmark_count} landmarks ({self.landmark_method}) "
                f"from {len(X_work)} points"
            )
            idx = self._select_landmarks(
                X_work, self.landmark_count, self.landmark_method
            )
            self._landmark_indices = idx
            self._landmarks = X_work[idx]
            if self.use_witness_complex:
                logger.info(
                    f"[ER-PH] Building witness complex on {len(self._landmarks)} landmarks "
                    f"with {len(X_work)} witnesses (k={self.witness_k})"
                )
                dist_sparse, assignments = self._witness_sparse_graph(
                    self._landmarks,
                    X_work,
                    k=self.witness_k,
                    batch_size=self.witness_batch_size,
                    metric=self.input_metric,
                    knn_k=self.sparse_knn_k or self.k,
                )
                self._landmark_assignments = assignments
                self._distance_sparse = dist_sparse
                self._distance = None
            else:
                logger.info(
                    f"[ER-PH] Computing effective resistance on landmarks "
                    f"(k={min(self.k, max(1, len(self._landmarks) - 1))})"
                )
                dist = effective_resistance_distance(
                    self._landmarks,
                    k=min(self.k, max(1, len(self._landmarks) - 1)),
                    corrected=self.corrected,
                    weighted=self.weighted,
                    backend=self.backend,
                    input_metric=self.input_metric,
                    torch_chunk_size=self.torch_chunk_size,
                    torch_device=self.torch_device,
                )
                # Map each subsample point to its nearest landmark for later hole sizing.
                self._landmark_assignments = self._assign_points_to_landmarks(
                    X_work, self._landmarks, metric=self.input_metric
                )
                self._distance_sparse = None
        else:
            self._landmark_indices = None
            self._landmarks = None
            logger.info(
                f"[ER-PH] Computing effective resistance on full data (k={self.k})"
            )
            dist = effective_resistance_distance(
                X_work,
                k=self.k,
                corrected=self.corrected,
                weighted=self.weighted,
                backend=self.backend,
                input_metric=self.input_metric,
                torch_chunk_size=self.torch_chunk_size,
                torch_device=self.torch_device,
            )
            self._distance_sparse = None
            self._landmark_assignments = None
        if self._distance_sparse is not None:
            return self._distance_sparse
        self._distance = dist
        return dist

    def detect_holes(self) -> EffectiveResistancePHResult:
        if self.X is None:
            raise RuntimeError("Call fit(X) before detect_holes().")

        if self._distance_sparse is not None:
            dist = self._distance_sparse
        elif self._distance is not None:
            dist = self._distance
        else:
            dist = self._compute_distance()

        logger.info(f"[ER-PH] Running ripser (maxdim={self.maxdim})")
        ph = ripser(
            dist,
            distance_matrix=True,
            maxdim=self.maxdim,
            do_cocycles=True,
        )
        diagrams: List[np.ndarray] = ph["dgms"]
        cocycles: List[List[np.ndarray]] = ph.get("cocycles", [])

        if len(diagrams) <= 1 or diagrams[1].size == 0:
            logger.info("No H1 features detected in persistent homology.")
            result = EffectiveResistancePHResult(
                persistence_intervals=np.empty((0, 2)),
                hole_centers=np.empty((0, self.X.shape[1])),
                hole_sizes=np.empty((0,), dtype=int),
                hole_sizes_subsample=np.empty((0,), dtype=int),
                hole_samples=np.empty((0, self.X.shape[1])),
                hole_sample_groups=[],
                hole_subsample_indices=[],
                filtered_holes=np.empty((0, self.X.shape[1])),
                cluster_labels=np.empty(0, dtype=int),
                maxima=np.empty((0, self.X.shape[1])),
                component_boundary_points=[],
                component_boundary_indices=[],
                component_boundary_distances=[],
                component_volumes=np.empty(0, dtype=float),
            )
            self._assign_result_attrs(result)
            return result

        h1 = diagrams[1]
        persistence = h1[:, 1] - h1[:, 0]
        keep = persistence >= self.min_persistence
        kept_indices = np.where(keep)[0]
        if len(h1):
            top = np.argsort(persistence)[::-1][:5]
            top_vals = np.array2string(persistence[top], precision=4)
            logger.info(f"[ER-PH] Top H1 persistences: {top_vals}")
        if self.max_holes is not None:
            order = np.argsort(persistence[kept_indices])[::-1][: self.max_holes]
            kept_indices = kept_indices[order]

        hole_centers: List[np.ndarray] = []
        hole_sample_groups: List[np.ndarray] = []
        hole_sizes: List[int] = []
        hole_sizes_subsample: List[int] = []
        hole_subsample_indices: List[np.ndarray] = []
        boundary_points: List[np.ndarray] = []
        boundary_indices: List[np.ndarray] = []
        boundary_distances: List[np.ndarray] = []
        component_volumes: List[float] = []

        logger.info(
            f"[ER-PH] Extracting {len(kept_indices)} candidate holes "
            f"(persistence filter {self.min_persistence:.4f})"
        )
        for idx in (
            tqdm(kept_indices, desc="Extracting holes", leave=False)
            if len(kept_indices) > 0
            else []
        ):
            if len(cocycles) <= 1 or idx >= len(cocycles[1]):
                continue
            cocycle = cocycles[1][idx]
            nodes = np.unique(cocycle[:, :2].astype(int))
            samples = (self._landmarks if self._landmarks is not None else self.X)[
                nodes
            ]
            hole_sample_groups.append(samples)
            hole_sizes.append(len(nodes))
            subsample_idx = self._points_for_nodes(nodes)
            hole_subsample_indices.append(subsample_idx)
            hole_sizes_subsample.append(len(subsample_idx))
            hole_centers.append(
                samples.mean(axis=0) if len(samples) else np.zeros(self.X.shape[1])
            )
            boundary, b_idx, b_dist = self._sample_boundary(samples)
            boundary_points.append(boundary)
            boundary_indices.append(b_idx)
            boundary_distances.append(b_dist)
            component_volumes.append(self._volume_proxy(samples))

        hole_samples = (
            np.vstack(hole_sample_groups)
            if hole_sample_groups
            else np.empty((0, self.X.shape[1]))
        )
        hole_centers_arr = (
            np.vstack(hole_centers) if hole_centers else np.empty((0, self.X.shape[1]))
        )
        hole_sizes_arr = np.asarray(hole_sizes, dtype=int)
        hole_sizes_subsample_arr = np.asarray(hole_sizes_subsample, dtype=int)
        intervals = h1[kept_indices[: len(hole_centers_arr)]]

        result = EffectiveResistancePHResult(
            persistence_intervals=intervals,
            hole_centers=hole_centers_arr,
            hole_sizes=hole_sizes_arr,
            hole_sizes_subsample=hole_sizes_subsample_arr,
            hole_samples=hole_samples,
            hole_sample_groups=hole_sample_groups,
            hole_subsample_indices=hole_subsample_indices,
            filtered_holes=hole_centers_arr,
            cluster_labels=(
                np.repeat(
                    np.arange(len(hole_sample_groups)),
                    [len(g) for g in hole_sample_groups],
                )
                if hole_sample_groups
                else np.empty(0, dtype=int)
            ),
            maxima=hole_centers_arr,
            component_boundary_points=boundary_points,
            component_boundary_indices=boundary_indices,
            component_boundary_distances=boundary_distances,
            component_volumes=np.asarray(component_volumes, dtype=float),
        )
        self._assign_result_attrs(result)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _witness_distance_matrix(
        self,
        landmarks: np.ndarray,
        witnesses: np.ndarray,
        *,
        k: int,
        batch_size: int,
        metric: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a landmark distance matrix using a strong-witness construction."""
        n_landmarks = len(landmarks)
        nn = NearestNeighbors(n_neighbors=min(k, n_landmarks), metric=metric)
        nn.fit(landmarks)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        assignments = np.empty(len(witnesses), dtype=int)

        for start in range(0, len(witnesses), batch_size):
            end = min(len(witnesses), start + batch_size)
            chunk = witnesses[start:end]
            dists, idxs = nn.kneighbors(chunk)
            assignments[start:end] = idxs[:, 0]
            for r in range(len(chunk)):
                inds = idxs[r]
                dist_row = dists[r]
                for i in range(len(inds)):
                    for j in range(i + 1, len(inds)):
                        a, b = int(inds[i]), int(inds[j])
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        rows.append(a)
                        cols.append(b)
                        data.append(float(max(dist_row[i], dist_row[j])))

        dist_mat = np.full((n_landmarks, n_landmarks), np.inf, dtype=float)
        np.fill_diagonal(dist_mat, 0.0)
        if rows:
            rows_arr = np.asarray(rows, dtype=int)
            cols_arr = np.asarray(cols, dtype=int)
            data_arr = np.asarray(data, dtype=float)
            key = rows_arr * n_landmarks + cols_arr
            order = np.argsort(key)
            key_sorted = key[order]
            rows_sorted = rows_arr[order]
            cols_sorted = cols_arr[order]
            data_sorted = data_arr[order]
            _, first_idx = np.unique(key_sorted, return_index=True)
            ends = np.append(first_idx[1:], len(data_sorted))
            mins = np.fromiter(
                (
                    data_sorted[first_idx[i] : ends[i]].min()
                    for i in range(len(first_idx))
                ),
                dtype=float,
                count=len(first_idx),
            )
            r_unique = rows_sorted[first_idx]
            c_unique = cols_sorted[first_idx]
            dist_mat[r_unique, c_unique] = mins
            dist_mat[c_unique, r_unique] = mins

        finite_mask = np.isfinite(dist_mat)
        if not finite_mask.any():
            raise RuntimeError("Witness distance matrix contains no finite entries.")
        max_finite = dist_mat[finite_mask].max()
        dist_mat[~finite_mask] = max_finite * 2.0

        return dist_mat, assignments

    def _assign_points_to_landmarks(
        self, points: np.ndarray, landmarks: np.ndarray, metric: str
    ) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=1, metric=metric)
        nn.fit(landmarks)
        _, idxs = nn.kneighbors(points)
        return idxs[:, 0].astype(int)

    def _points_for_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Return indices in the subsampled X that are assigned to the given landmarks."""
        if self.X is None:
            return np.empty((0,), dtype=int)
        if self._landmark_assignments is None or self._subsample_indices is None:
            return nodes
        mask = np.isin(self._landmark_assignments, nodes)
        return self._subsample_indices[mask]

    def _witness_sparse_graph(
        self,
        landmarks: np.ndarray,
        witnesses: np.ndarray,
        *,
        k: int,
        batch_size: int,
        metric: str,
        knn_k: int,
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build a sparse landmark graph using landmark kNN and strong-witness edges."""
        n_landmarks = len(landmarks)
        nn = NearestNeighbors(
            n_neighbors=min(knn_k, max(1, n_landmarks)),
            metric=metric,
        )
        nn.fit(landmarks)

        edge_weights: dict[Tuple[int, int], float] = {}

        def add_edge(a: int, b: int, w: float) -> None:
            if a == b:
                return
            if a > b:
                a, b = b, a
            prev = edge_weights.get((a, b))
            if prev is None or w < prev:
                edge_weights[(a, b)] = w

        # Landmark kNN edges
        lm_dists, lm_idxs = nn.kneighbors(landmarks)
        for i in range(n_landmarks):
            for nb, w in zip(lm_idxs[i], lm_dists[i]):
                add_edge(int(i), int(nb), float(w))

        # Witness-induced edges (strong witness: pairwise among k nearest landmarks)
        assignments = np.empty(len(witnesses), dtype=int)
        for start in range(0, len(witnesses), batch_size):
            end = min(len(witnesses), start + batch_size)
            chunk = witnesses[start:end]
            dists, idxs = nn.kneighbors(chunk)
            assignments[start:end] = idxs[:, 0]
            for r in range(len(chunk)):
                inds = idxs[r]
                dist_row = dists[r]
                for i in range(len(inds)):
                    for j in range(i + 1, len(inds)):
                        a, b = int(inds[i]), int(inds[j])
                        w = float(max(dist_row[i], dist_row[j]))
                        add_edge(a, b, w)

        if edge_weights:
            rows, cols, data = zip(*((a, b, w) for (a, b), w in edge_weights.items()))
            rows_list = list(rows) + list(cols)
            cols_list = list(cols) + list(rows)
            data_list = list(data) + list(data)
            dist_sparse = sp.coo_matrix(
                (data_list, (rows_list, cols_list)),
                shape=(n_landmarks, n_landmarks),
            ).tocsr()
        else:
            dist_sparse = sp.csr_matrix((n_landmarks, n_landmarks), dtype=float)

        dist_sparse.setdiag(0.0)
        return dist_sparse, assignments

    def _assign_result_attrs(self, result: EffectiveResistancePHResult) -> None:
        """Expose result fields on the detector for downstream metrics/export."""
        self.hole_centers = result.hole_centers
        self.hole_sizes = result.hole_sizes
        self.hole_sizes_subsample = result.hole_sizes_subsample
        self.hole_samples = result.hole_samples
        self.hole_sample_groups = result.hole_sample_groups
        self.hole_subsample_indices = result.hole_subsample_indices
        self.filtered_holes = result.filtered_holes
        self.cluster_labels = result.cluster_labels
        self.maxima = result.maxima
        self.component_boundary_points = result.component_boundary_points
        self.component_boundary_indices = result.component_boundary_indices
        self.component_boundary_distances = result.component_boundary_distances
        self.component_volumes = result.component_volumes

    def _sample_boundary(
        self, samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if samples.size == 0 or self.X is None:
            return (
                np.empty((0, self.X.shape[1])),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )
        nn = NearestNeighbors(
            n_neighbors=min(self.boundary_neighbor_k, len(self.X)),
            algorithm="auto",
        )
        nn.fit(self.X)
        dists, idxs = nn.kneighbors(samples)
        radius = np.median(dists) * self.boundary_radius_scale
        mask = dists <= radius
        boundary_idx = np.unique(idxs[mask])
        if boundary_idx.size == 0:
            return (
                np.empty((0, self.X.shape[1])),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=float),
            )
        # For each boundary index, keep the minimum observed distance within the mask.
        flat_idx = idxs[mask].ravel()
        flat_dists = dists[mask].ravel()
        order = np.argsort(flat_idx)
        flat_idx = flat_idx[order]
        flat_dists = flat_dists[order]
        uniq, start = np.unique(flat_idx, return_index=True)
        end = np.append(start[1:], len(flat_idx))
        min_dists = np.fromiter(
            (flat_dists[start[i] : end[i]].min() for i in range(len(start))),
            dtype=float,
            count=len(start),
        )
        return self.X[uniq], uniq, min_dists

    def _volume_proxy(self, samples: np.ndarray) -> float:
        if len(samples) < 2:
            return 0.0
        # Use median distance to centroid to define a radius; avoids O(n^2) memory
        centroid = samples.mean(axis=0)
        dists = np.linalg.norm(samples - centroid, axis=1)
        r = float(np.median(dists))
        if r <= 0:
            return 0.0
        d = samples.shape[1]
        try:
            vol = (np.pi ** (d / 2) / np.math.gamma(d / 2 + 1)) * (r**d)
        except Exception:
            vol = 0.0
        return vol

    def _select_landmarks(self, X: np.ndarray, m: int, method: str) -> np.ndarray:
        if m >= len(X):
            return np.arange(len(X))
        if method == "random":
            rng = np.random.default_rng(0)
            return rng.choice(len(X), size=m, replace=False)
        if method == "fps":
            rng = np.random.default_rng(0)
            first = rng.integers(0, len(X))
            landmarks = [int(first)]
            dist = np.linalg.norm(X - X[first], axis=1)
            for _ in range(m - 1):
                idx = int(np.argmax(dist))
                landmarks.append(idx)
                dist = np.minimum(dist, np.linalg.norm(X - X[idx], axis=1))
            return np.array(landmarks, dtype=int)
        raise ValueError("landmark_method must be 'random' or 'fps'.")

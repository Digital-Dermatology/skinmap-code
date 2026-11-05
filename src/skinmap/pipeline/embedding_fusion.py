"""
Embedding Fusion Module: Lean Concat → Projector → InfoNCE

This module implements the minimal, motivated pipeline to fuse multiple teacher encoders
into a single, shared image-text embedding using a small projector trained with CLIP-style InfoNCE.

Based on the better_embeddings.md specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import eigh
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

# =============================
# Whitening & PCA Utilities
# =============================


def fit_whitener(
    Z: np.ndarray, eps: float = 1e-5, max_components: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-model whitening on rows of Z (N x d).

    Returns (mu, W) where W @ (z - mu) whitens (approximately identity covariance).

    Args:
        Z: Embedding matrix (N x d) where N is number of samples
        eps: Small constant for numerical stability
        max_components: Maximum number of components to compute (default: None, use all).
                       If set, uses truncated eigendecomposition for speed.

    Returns:
        mu: Mean vector (1 x d)
        W: Whitening matrix (d x d or k x d if max_components is set)
    """
    assert Z.ndim == 2, f"Expected 2D array, got shape {Z.shape}"

    # Convert to float32 if needed (numpy/scipy operations don't support float16)
    if Z.dtype == np.float16:
        Z = Z.astype(np.float32)

    mu = Z.mean(axis=0, keepdims=True)
    X = Z - mu

    N, d = X.shape

    # Use SVD-based whitening for high dimensions or when max_components is set
    # This is much faster than covariance eigendecomposition when N < d or we only need top components
    use_svd = (N < d * 2) or (max_components is not None and max_components < d)

    if use_svd:
        # SVD approach: X = U @ S @ V.T
        # This is faster when N < d and we don't need all components
        if max_components is not None:
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(
                n_components=min(max_components, min(N, d) - 1), random_state=42
            )
            svd.fit(X)
            # Get singular values and components
            singular_values = svd.singular_values_
            components = svd.components_  # (k, d)

            # Convert to eigenvalues of covariance: eigenval = (s^2) / (N-1)
            eigenvalues = (singular_values**2) / max(1, N - 1)
            eigenvalues = np.clip(eigenvalues, eps, None)

            # W = D^{-1/2} @ V^T where V are the right singular vectors (components)
            inv_sqrt_diag = 1.0 / np.sqrt(eigenvalues)
            W = np.diag(inv_sqrt_diag) @ components
        else:
            # Full SVD (still faster than covariance for N < d)
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            eigenvalues = (s**2) / max(1, N - 1)
            eigenvalues = np.clip(eigenvalues, eps, None)
            inv_sqrt_diag = 1.0 / np.sqrt(eigenvalues)
            W = np.diag(inv_sqrt_diag) @ Vt
    else:
        # Covariance approach (better when N >> d and we need all components)
        C = (X.T @ X) / max(1, N - 1)

        # Eigen-decompose using scipy's faster divide-and-conquer algorithm
        # driver='evd' uses LAPACK's divide-and-conquer (2-5x faster than default QR)
        vals, vecs = eigh(C, driver="evd")
        vals = np.clip(vals, eps, None)
        inv_sqrt = np.diag(1.0 / np.sqrt(vals))
        # Whitening transformation: (z - mu) @ W.T should give identity covariance
        # Since C = V @ D @ V^T, we want W.T = V @ D^{-1/2}
        # Therefore W = D^{-1/2} @ V^T
        W = inv_sqrt @ vecs.T

    return mu.astype(np.float32), W.astype(np.float32)


def apply_whiten(z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply whitening transformation to embedding(s).

    Args:
        z: Embedding(s) to whiten (can be 1D or 2D)
        mu: Mean vector from fit_whitener
        W: Whitening matrix from fit_whitener

    Returns:
        Whitened embedding(s)
    """
    return (z - mu) @ W.T


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize along specified axis.

    Args:
        x: Array to normalize
        axis: Axis along which to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized array
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def fit_pca(X: np.ndarray, D: int) -> np.ndarray:
    """Return top-D principal components (D x D_in).

    Uses SVD on centered data; components are row vectors.

    Args:
        X: Data matrix (N x D_in)
        D: Number of components to keep

    Returns:
        PCA components matrix (D x D_in)
    """
    # Convert to float32 if needed (numpy SVD doesn't support float16)
    if X.dtype == np.float16:
        X = X.astype(np.float32)

    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:D, :]
    return comps.astype(np.float32)


# =============================
# Dataset & Sampler
# =============================


@dataclass
class Sample:
    """Single sample with precomputed embeddings."""

    image_id: str
    text_vec: np.ndarray  # (d_text,) from CLIP text tower, L2-normalized
    source: str  # for domain-balanced sampling
    teacher_embs: Dict[str, np.ndarray]  # name -> (d_i,) raw or L2-normalized


class PrecomputedDataset(Dataset):
    """Dataset for precomputed teacher embeddings."""

    def __init__(self, samples: List[Sample], teacher_order: List[str]):
        self.samples = samples
        self.teacher_order = teacher_order

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        embs = [s.teacher_embs[name] for name in self.teacher_order]
        return {
            "image_id": s.image_id,
            "teacher_embs": embs,  # list of np arrays
            "text_vec": s.text_vec,  # np array
            "source": s.source,
        }


class DomainBalancedBatchSampler(Sampler[List[int]]):
    """Simple domain-balanced sampler.

    Groups indices by `source` and cycles to build balanced batches.
    """

    def __init__(self, sources: List[str], batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by source
        self.by_src: Dict[str, List[int]] = {}
        for idx, src in enumerate(sources):
            self.by_src.setdefault(src, []).append(idx)

        # Shuffle within source if requested
        if shuffle:
            for k in self.by_src:
                rng = np.random.default_rng()
                rng.shuffle(self.by_src[k])

        self.sources = list(self.by_src.keys())
        self.ptrs = {k: 0 for k in self.sources}

    def __iter__(self):
        # Reset pointers at the start of each epoch
        self.ptrs = {k: 0 for k in self.sources}

        # Reshuffle within each source if requested
        if self.shuffle:
            for k in self.by_src:
                rng = np.random.default_rng()
                rng.shuffle(self.by_src[k])

        per = max(1, self.batch_size // max(1, len(self.sources)))
        n_batches = sum(len(v) for v in self.by_src.values()) // self.batch_size

        for _ in range(n_batches):
            batch = []
            for src in self.sources:
                start = self.ptrs[src]
                end = min(start + per, len(self.by_src[src]))
                if start >= end:
                    continue
                batch.extend(self.by_src[src][start:end])
                self.ptrs[src] = end

            # Top up from any source if needed
            if len(batch) < self.batch_size:
                for src in self.sources:
                    while len(batch) < self.batch_size and self.ptrs[src] < len(
                        self.by_src[src]
                    ):
                        batch.append(self.by_src[src][self.ptrs[src]])
                        self.ptrs[src] += 1

            if batch:
                yield batch

    def __len__(self):
        total = sum(len(v) for v in self.by_src.values())
        return max(1, total // self.batch_size)


# =============================
# Projector Modules
# =============================


class LinearProjector(nn.Module):
    """Linear projection with L2 normalization."""

    def __init__(self, d_in: int, d_out: int, init_matrix: Optional[np.ndarray] = None):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)

        if init_matrix is not None:
            assert init_matrix.shape == (
                d_out,
                d_in,
            ), f"Init matrix shape {init_matrix.shape} != expected ({d_out}, {d_in})"
            with torch.no_grad():
                self.proj.weight.copy_(torch.from_numpy(init_matrix))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        y = nn.functional.normalize(y, dim=-1)
        return y


class MLPProjector(nn.Module):
    """2-layer MLP projection with GELU activation and L2 normalization."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hid: Optional[int] = None,
        init_first: Optional[np.ndarray] = None,
    ):
        super().__init__()
        d_hid = d_hid or (2 * d_out)

        self.lin1 = nn.Linear(d_in, d_hid, bias=True)
        self.lin2 = nn.Linear(d_hid, d_out, bias=False)

        if init_first is not None:
            with torch.no_grad():
                # Use top components for initialization (up to d_hid rows)
                # init_first is (D, d_in) from PCA, we need (d_hid, d_in) for lin1
                n_components = min(d_hid, init_first.shape[0])
                if n_components < d_hid:
                    # PCA has fewer components than d_hid, init only part of the weights
                    self.lin1.weight[:n_components, :].copy_(
                        torch.from_numpy(init_first[:n_components, :])
                    )
                    # The rest of the weights will use default initialization
                else:
                    # We have enough PCA components
                    self.lin1.weight.copy_(torch.from_numpy(init_first[:d_hid, :]))

        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.gelu(self.lin1(x))
        y = self.lin2(h)
        y = nn.functional.normalize(y, dim=-1)
        return y


class ImageTextModel(nn.Module):
    """Dual projector model for image and text embeddings with learnable temperature."""

    def __init__(
        self,
        d_cat: int,
        d_text: int,
        d_out: int,
        img_init: Optional[np.ndarray] = None,
        txt_init: Optional[np.ndarray] = None,
        kind: str = "linear",
    ):
        super().__init__()

        if kind == "linear":
            self.img_head = LinearProjector(d_cat, d_out, img_init)
        elif kind == "mlp":
            self.img_head = MLPProjector(
                d_cat, d_out, d_hid=2 * d_out, init_first=img_init
            )
        else:
            raise ValueError(f"Unknown projector kind: {kind}")

        self.txt_head = LinearProjector(d_text, d_out, txt_init)

        # Learnable temperature as in CLIP (start around 0.07)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(
        self, z_cat: torch.Tensor, t_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            z_cat: Concatenated image embeddings (B, d_cat)
            t_vec: Text embeddings (B, d_text)

        Returns:
            zi: Projected image embeddings (B, d_out)
            zt: Projected text embeddings (B, d_out)
            logit_scale: Scalar temperature parameter
        """
        zi = self.img_head(z_cat)  # [B, D]
        zt = self.txt_head(t_vec)  # [B, D]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return zi, zt, logit_scale


# =============================
# InfoNCE Loss
# =============================


def info_nce_logits(
    zi: torch.Tensor, zt: torch.Tensor, logit_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute InfoNCE logits for image and text.

    Args:
        zi: Image embeddings (B, D)
        zt: Text embeddings (B, D)
        logit_scale: Temperature parameter

    Returns:
        logits_per_image: (B, B)
        logits_per_text: (B, B)
        labels: (B,) with diagonal indices
    """
    logits_per_image = logit_scale * (zi @ zt.T)
    logits_per_text = logits_per_image.T
    labels = torch.arange(zi.size(0), device=zi.device)
    return logits_per_image, logits_per_text, labels


def clip_loss(
    zi: torch.Tensor, zt: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    """Compute symmetric CLIP InfoNCE loss.

    Args:
        zi: Image embeddings (B, D)
        zt: Text embeddings (B, D)
        logit_scale: Temperature parameter

    Returns:
        Loss scalar
    """
    lip, ltp, labels = info_nce_logits(zi, zt, logit_scale)
    loss_i = nn.functional.cross_entropy(lip, labels)
    loss_t = nn.functional.cross_entropy(ltp, labels)
    return 0.5 * (loss_i + loss_t)


# =============================
# Collate Function with Whitening
# =============================


@dataclass
class WhitenSpec:
    """Specification for whitening transformations."""

    mu: List[np.ndarray]  # per-teacher [1, d_i]
    W: List[np.ndarray]  # per-teacher [d_i, d_i]
    dims: List[int]
    gates: Optional[List[float]] = None  # optional scalar per-teacher


def make_collate(whiten: WhitenSpec):
    """Create a collate function that applies whitening."""
    mu_t = [torch.from_numpy(m).float().squeeze(0) for m in whiten.mu]
    W_t = [torch.from_numpy(w).float() for w in whiten.W]
    gates = whiten.gates or [1.0] * len(mu_t)

    def collate(batch: List[dict]):
        # teacher_embs: list of np arrays
        imgs = []
        for item in batch:
            blocks = []
            for j, emb in enumerate(item["teacher_embs"]):
                z = torch.from_numpy(emb).float()
                z = z / (z.norm(p=2) + 1e-8)
                z = (z - mu_t[j]) @ W_t[j].T
                if gates[j] != 1.0:
                    z = z * float(gates[j])
                blocks.append(z)
            z_cat = torch.cat(blocks, dim=-1)
            imgs.append(z_cat)
        Z = torch.stack(imgs, dim=0)  # [B, D_cat]

        T = torch.stack(
            [torch.from_numpy(x["text_vec"]).float() for x in batch], dim=0
        )  # [B, d_text]

        return {"z_cat": Z, "t_vec": T, "source": [x["source"] for x in batch]}

    return collate


# =============================
# Training Loop
# =============================


def train_one_epoch(
    model: ImageTextModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
    structure_loss=None,
    wandb_run=None,
    log_prefix: str = "train",
    epoch: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch.

    Args:
        model: ImageTextModel to train
        loader: DataLoader with collate function applied
        optimizer: Optimizer
        device: Device to train on
        structure_loss: Optional CLIPLoss with structure regularization

    Returns:
        Average loss for the epoch, and dict of auxiliary losses
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    aux_totals: Dict[str, float] = {}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        z_cat = batch["z_cat"].to(device, non_blocking=True)
        t_vec = batch["t_vec"].to(device, non_blocking=True)

        zi, zt, ls = model(z_cat, t_vec)

        if structure_loss is not None:
            structure_loss.step()
            loss_dict = structure_loss(
                zi,
                zt,
                z_cat,
                t_vec,
                image_embeddings_aligned_alt=None,
            )
            loss = loss_dict["overall_loss"]
            for key, value in loss_dict.items():
                if key == "overall_loss":
                    continue
                aux_totals[key] = aux_totals.get(key, 0.0) + float(value.item())
        else:
            loss = clip_loss(zi, zt, ls)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss * z_cat.size(0)
        total_samples += z_cat.size(0)

        if wandb_run is not None:
            logit_scale = getattr(model, "logit_scale", None)
            effective_temp = None
            if logit_scale is not None:
                scale_val = logit_scale.exp().detach().cpu().item()
                effective_temp = 1.0 / max(scale_val, 1e-8)

            batch_metrics = {
                f"{log_prefix}/batch_loss": batch_loss,
                f"{log_prefix}/batch": batch_idx,
                f"{log_prefix}/epoch": epoch,
            }
            if effective_temp is not None:
                batch_metrics[f"{log_prefix}/logit_scale"] = scale_val
                batch_metrics[f"{log_prefix}/temperature"] = effective_temp
            if structure_loss is not None:
                for key, value in loss_dict.items():
                    if key == "overall_loss":
                        continue
                    batch_metrics[f"{log_prefix}/{key}"] = float(value.item())
            wandb_run.log(batch_metrics)

    avg_loss = total_loss / max(1, total_samples)
    avg_aux = {k: v / max(1, len(loader)) for k, v in aux_totals.items()}
    return avg_loss, avg_aux


@torch.no_grad()
def evaluate_recall_at_k(
    model: ImageTextModel,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    """Evaluate retrieval recall@k.

    Args:
        model: ImageTextModel to evaluate
        loader: DataLoader
        device: Device
        ks: Tuple of k values to compute recall for

    Returns:
        Dictionary mapping k -> recall@k
    """
    model.eval()
    ims, txts = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        z_cat = batch["z_cat"].to(device, non_blocking=True)
        t_vec = batch["t_vec"].to(device, non_blocking=True)
        zi, zt, _ = model(z_cat, t_vec)
        ims.append(zi.cpu())
        txts.append(zt.cpu())

    ims = torch.cat(ims, dim=0)
    txts = torch.cat(txts, dim=0)
    sims = ims @ txts.T
    ranks = sims.argsort(dim=1, descending=True)

    metrics = {}
    for k in ks:
        topk = ranks[:, :k]
        labels = torch.arange(ims.size(0)).unsqueeze(1)
        r_at_k = (topk == labels).any(dim=1).float().mean().item()
        metrics[k] = r_at_k

    return metrics


# =============================
# Build Specification
# =============================


@dataclass
class BuildSpec:
    """Specification for building the model."""

    teacher_names: List[str]
    teacher_dims: List[int]
    text_dim: int
    out_dim: int
    pca_image: Optional[np.ndarray] = None  # [D, D_cat], optional init
    pca_text: Optional[np.ndarray] = None  # [D, d_text], optional init
    kind: str = "linear"  # or "mlp"


def build_model(spec: BuildSpec) -> ImageTextModel:
    """Build ImageTextModel from specification.

    Args:
        spec: BuildSpec with model configuration

    Returns:
        Initialized ImageTextModel
    """
    d_cat = int(sum(spec.teacher_dims))
    return ImageTextModel(
        d_cat=d_cat,
        d_text=spec.text_dim,
        d_out=spec.out_dim,
        img_init=spec.pca_image,
        txt_init=spec.pca_text,
        kind=spec.kind,
    )

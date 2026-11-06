"""Synthetic datasets used to evaluate hole detection."""

from __future__ import annotations

from typing import Optional

import numpy as np


def make_ring(
    n: int = 3000,
    radius: float = 3.0,
    noise: float = 0.15,
    random_state: Optional[int] = None,
    dim: int = 2,
) -> np.ndarray:
    """Sample a noisy ring in 2D (or embed in higher dims if dim>2)."""
    if dim < 2:
        raise ValueError("Ring requires dim >= 2.")
    rng = np.random.default_rng(random_state)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = radius + noise * rng.standard_normal(size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.zeros((n, dim), dtype=float)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def sample_hypersphere_shell(
    n: int,
    d: int,
    r_in: float = 0.8,
    r_out: float = 1.6,
    noise: float = 0.02,
    center: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample points uniformly inside a hollow hyperspherical shell."""
    rng = np.random.default_rng(seed)

    pts = rng.standard_normal((n, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    u = rng.random(n)
    radii = (r_in**d + u * (r_out**d - r_in**d)) ** (1.0 / d)
    pts *= radii[:, None]

    if center is not None:
        pts += np.asarray(center)

    if noise > 0:
        pts += noise * rng.standard_normal(pts.shape)

    return pts


def sample_c_shape_nd(
    n: int,
    d: int,
    r_in: float = 0.8,
    r_out: float = 1.6,
    gap_fraction: float = 0.25,
    noise: float = 0.02,
    center: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a hyperspherical shell with a wedge removed (generalised C-shape)."""
    rng = np.random.default_rng(seed)

    oversample = int(n / max(1e-6, 1 - gap_fraction)) + 1000
    pts = rng.standard_normal((oversample, d))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    u = rng.random(oversample)
    radii = (r_in**d + u * (r_out**d - r_in**d)) ** (1.0 / d)
    pts *= radii[:, None]

    threshold = np.cos(2 * np.pi * gap_fraction)
    if d == 1:
        keep_mask = pts[:, 0] < threshold
    else:
        angle_coords = pts[:, :2]
        norms = np.linalg.norm(angle_coords, axis=1)
        normalized = angle_coords / (norms[:, None] + 1e-10)
        keep_mask = normalized[:, 0] < threshold

    pts = pts[keep_mask][:n]

    if center is not None:
        pts += np.asarray(center)
    if noise > 0:
        pts += noise * rng.standard_normal(pts.shape)

    return pts


def sample_two_shells(
    n: int,
    d: int,
    separation: float = 6.0,
    r_in: float = 0.8,
    r_out: float = 1.6,
    noise: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample two separated hyperspherical shells that create three holes."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    center1 = np.zeros(d)
    center2 = np.zeros(d)
    center2[0] = separation

    shell1 = sample_hypersphere_shell(
        n1, d, r_in, r_out, noise, center1, rng.integers(0, 1_000_000)
    )
    shell2 = sample_hypersphere_shell(
        n2, d, r_in, r_out, noise, center2, rng.integers(0, 1_000_000)
    )
    return np.vstack([shell1, shell2])


def sample_two_rings_small(
    n: int,
    d: int,
    separation: float = 3.0,
    large_r_in: float = 0.8,
    large_r_out: float = 1.6,
    scale_small: float = 0.5,
    noise: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample one large shell and a smaller offset shell (generalised 'two rings')."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    center_large = np.zeros(d)
    center_small = np.zeros(d)
    center_small[0] = separation

    large = sample_hypersphere_shell(
        n1,
        d,
        r_in=large_r_in,
        r_out=large_r_out,
        noise=noise,
        center=center_large,
        seed=rng.integers(0, 1_000_000),
    )
    small = sample_hypersphere_shell(
        n2,
        d,
        r_in=large_r_in * scale_small,
        r_out=large_r_out * scale_small,
        noise=noise,
        center=center_small,
        seed=rng.integers(0, 1_000_000),
    )
    return np.vstack([large, small])


def sample_shell_with_blob(
    n: int,
    d: int,
    blob_fraction: float = 0.15,
    blob_location: str = "center",
    noise: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a shell plus a Gaussian blob (useful for false-positive checks)."""
    rng = np.random.default_rng(seed)

    n_shell = max(1, int(n * (1 - blob_fraction)))
    n_blob = max(0, n - n_shell)

    shell = sample_hypersphere_shell(
        n_shell,
        d,
        r_in=0.8,
        r_out=1.6,
        noise=noise,
        center=None,
        seed=rng.integers(0, 1_000_000),
    )

    if blob_location == "center":
        blob_center = np.zeros(d)
    elif blob_location == "offset":
        blob_center = np.zeros(d)
        blob_center[0] = 1.5
    else:
        blob_center = np.zeros(d)

    if n_blob > 0:
        blob = rng.standard_normal((n_blob, d)) * 0.3 + blob_center
        return np.vstack([shell, blob])
    return shell

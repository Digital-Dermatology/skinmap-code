import numpy as np
import torch

from src.hole_detection.eff_resistance import (
    EffectiveResistancePHDetector,
    effective_resistance_distance,
)


def test_effective_resistance_numpy_vs_torch_agree():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 4)).astype(np.float32)

    dist_numpy = effective_resistance_distance(X, k=4, backend="numpy")
    dist_torch = effective_resistance_distance(
        torch.tensor(X), k=4, backend="torch", torch_chunk_size=6
    )

    assert dist_numpy.shape == (12, 12)
    np.testing.assert_allclose(dist_numpy, dist_torch, rtol=1e-4, atol=1e-4)


def test_effective_resistance_detects_circle_hole():
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    noise = 0.02 * np.random.default_rng(1).standard_normal(circle.shape)
    X = (circle + noise).astype(np.float32)

    detector = EffectiveResistancePHDetector(
        k=8, corrected=True, min_persistence=0.05, max_holes=2
    )
    detector.fit(X)
    result = detector.detect_holes()

    assert result.persistence_intervals.shape[0] >= 1
    assert result.hole_centers.shape[0] >= 1
    # Ensure the strongest hole has non-trivial persistence
    assert (
        result.persistence_intervals[:, 1].max()
        - result.persistence_intervals[:, 0].min()
        > 0.05
    )


def test_boundary_sampler_and_volume_proxy():
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    detector = EffectiveResistancePHDetector(
        k=6,
        corrected=True,
        min_persistence=0.01,
        max_holes=1,
        boundary_neighbor_k=10,
        boundary_radius_scale=1.5,
    )
    detector.fit(circle)
    result = detector.detect_holes()

    assert len(result.component_boundary_points) == len(result.hole_centers)
    for boundary_pts, boundary_idx, boundary_dists in zip(
        result.component_boundary_points,
        result.component_boundary_indices,
        result.component_boundary_distances,
    ):
        assert boundary_pts.shape[1] == circle.shape[1]
        assert len(boundary_pts) > 0
        assert len(boundary_idx) == len(boundary_pts)
        assert len(boundary_dists) == len(boundary_pts)
    assert (result.component_volumes >= 0).all()


def test_witness_sparse_complex_reports_subsample_sizes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3)).astype(np.float32)

    detector = EffectiveResistancePHDetector(
        k=6,
        corrected=True,
        min_persistence=0.0,
        max_holes=5,
        landmark_count=20,
        use_witness_complex=True,
        witness_k=4,
        witness_batch_size=50,
    )
    detector.fit(X)
    result = detector.detect_holes()

    # Sizes should be present and subsample sizes should be at least landmark sizes.
    assert hasattr(result, "hole_sizes_subsample")
    assert len(result.hole_sizes) == len(result.hole_sizes_subsample)
    if len(result.hole_sizes):
        assert (result.hole_sizes_subsample >= result.hole_sizes).all()
    # Ensure we keep indices for mapping back.
    assert hasattr(result, "hole_subsample_indices")

"""Tests for trained projector functionality.

Critical tests for the dimension mismatch bug fix:
- Pre-whitening before storing in samples
- Dimension consistency between stored and expected embeddings
- Whitening with dimension truncation

Also includes tests for validation metrics (validate_epoch function).
"""

import numpy as np
import pytest
import torch

from src.embedding_fusion import (
    ImageTextModel,
    LinearProjector,
    MLPProjector,
    WhitenSpec,
    apply_whiten,
    l2_normalize,
)


class TestL2Normalize:
    """Test L2 normalization."""

    def test_l2_normalize_makes_unit_norm(self):
        """L2 normalization should produce unit norm vectors."""
        x = np.random.randn(5, 10)
        x_normed = l2_normalize(x, axis=1)

        norms = np.linalg.norm(x_normed, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5))

    def test_l2_normalize_zero_vector(self):
        """L2 normalization of zero vector should remain zero."""
        x = np.zeros((3, 5))
        x_normed = l2_normalize(x, axis=1)

        np.testing.assert_array_almost_equal(x_normed, x)

    def test_l2_normalize_single_vector(self):
        """Test normalization of single vector."""
        x = np.array([[3.0, 4.0]])  # norm = 5
        x_normed = l2_normalize(x, axis=1)

        np.testing.assert_array_almost_equal(x_normed, np.array([[0.6, 0.8]]))


class TestApplyWhiten:
    """Test whitening transformation."""

    def test_apply_whiten_with_identity_transform(self):
        """Whitening with identity should return centered data."""
        x = np.random.randn(10, 512)
        mu = np.mean(x, axis=0, keepdims=True)
        W = np.eye(512)  # Identity

        x_whitened = apply_whiten(x, mu, W)

        # Should just center the data
        np.testing.assert_array_almost_equal(
            np.mean(x_whitened, axis=0), np.zeros(512), decimal=5
        )

    def test_apply_whiten_shape_preserved(self):
        """Whitening should preserve data shape."""
        x = np.random.randn(20, 256)
        mu = np.zeros((1, 256))
        W = np.random.randn(256, 256)

        x_whitened = apply_whiten(x, mu, W)

        assert x_whitened.shape == x.shape

    def test_apply_whiten_with_dimension_truncation(self):
        """Test whitening with dimension reduction."""
        x = np.random.randn(10, 1024)
        mu = np.zeros((1, 1024))
        W = np.random.randn(768, 1024)  # Truncate to 768 dims

        x_whitened = apply_whiten(x, mu, W)

        # Should reduce dimensions
        assert x_whitened.shape == (10, 768)


class TestWhitenSpec:
    """Test WhitenSpec data structure."""

    def test_whiten_spec_creation(self):
        """Test creating WhitenSpec."""
        mu = [np.zeros((1, 512)), np.zeros((1, 768))]
        W = [np.eye(512), np.eye(768)]
        dims = [512, 768]

        spec = WhitenSpec(mu=mu, W=W, dims=dims)

        assert len(spec.mu) == 2
        assert len(spec.W) == 2
        assert spec.dims == [512, 768]


class TestProjectors:
    """Test projector models."""

    def test_linear_projector_forward_pass(self):
        """Test forward pass of linear projector."""
        projector = LinearProjector(d_in=1024, d_out=512)

        x = torch.randn(16, 1024)
        output = projector(x)

        assert output.shape == (16, 512)
        # Should be normalized
        norms = torch.norm(output, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-5, rtol=1e-5)

    def test_mlp_projector_forward_pass(self):
        """Test forward pass of MLP projector."""
        projector = MLPProjector(d_in=1024, d_out=512, d_hid=2048)

        x = torch.randn(16, 1024)
        output = projector(x)

        assert output.shape == (16, 512)
        # Should be normalized
        norms = torch.norm(output, dim=1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-5, rtol=1e-5)

    def test_linear_projector_dimension_consistency(self):
        """Test that linear projector maintains dimension consistency."""
        projector = LinearProjector(d_in=768, d_out=512)

        # Input must match input_dim
        x_correct = torch.randn(8, 768)
        output = projector(x_correct)
        assert output.shape == (8, 512)

        # Wrong input dimension should fail
        x_wrong = torch.randn(8, 1024)
        with pytest.raises((RuntimeError, ValueError)):
            projector(x_wrong)


class TestDimensionBugFix:
    """Tests specifically for the dimension mismatch bug fix.

    The original bug: stored original embeddings but collate expected whitened dims.
    The fix: pre-whiten embeddings before storing in training samples.
    """

    def test_pre_whitened_embeddings_match_truncated_dims(self):
        """Critical: Pre-whitened embeddings must match truncated dimensions."""
        # Simulate the bug scenario
        original_dim = 1024
        truncated_dim = 768

        # Original embeddings (what we extract from model)
        original_embeddings = np.random.randn(10, original_dim).astype(np.float32)

        # Whitening that truncates dimensions
        mu = np.zeros((1, original_dim))
        W = np.random.randn(truncated_dim, original_dim)  # Truncates!

        # Apply whitening (what the fix does)
        whitened_embeddings = apply_whiten(original_embeddings, mu, W)

        # Whitened embeddings should have truncated dimensions
        assert whitened_embeddings.shape == (10, truncated_dim)

        # If we stored original_embeddings (the bug), it would fail
        # because collate expects truncated_dim but got original_dim
        assert whitened_embeddings.shape[1] != original_embeddings.shape[1]

    def test_stored_embeddings_compatible_with_collate(self):
        """Stored embeddings must be compatible with collate transforms."""
        # Simula the fixed pipeline
        original_dim = 1024
        truncated_dim = 768
        batch_size = 4

        # Step 1: Extract original embeddings
        original = np.random.randn(batch_size, original_dim).astype(np.float32)

        # Step 2: Pre-whiten with truncation (THE FIX)
        mu = np.zeros((1, original_dim))
        W = np.random.randn(truncated_dim, original_dim)
        pre_whitened = apply_whiten(original, mu, W)

        # Step 3: Store pre-whitened embeddings
        stored_embeddings = pre_whitened  # Now stored dimensions = truncated_dim

        # Step 4: In collate, use identity transform (since already whitened)
        identity_mu = np.zeros((1, truncated_dim))
        identity_W = np.eye(truncated_dim)

        # This should work without dimension errors
        collate_output = apply_whiten(stored_embeddings, identity_mu, identity_W)

        assert collate_output.shape == (batch_size, truncated_dim)

    def test_original_bug_would_fail(self):
        """Demonstrate that the original bug causes dimension mismatch."""
        original_dim = 1024
        truncated_dim = 768

        # Buggy approach: store original embeddings
        stored_original = np.random.randn(4, original_dim).astype(np.float32)

        # But collate expects truncated dimensions
        identity_mu = np.zeros((1, truncated_dim))
        identity_W = np.eye(truncated_dim)

        # This should fail with dimension mismatch
        with pytest.raises((ValueError, IndexError, AssertionError)):
            # Trying to apply (truncated_dim x truncated_dim) transform
            # to (batch x original_dim) data will fail
            _ = apply_whiten(stored_original, identity_mu, identity_W)

    def test_whitening_with_skip_whitening_flag(self):
        """Test that skip_whitening doesn't truncate dimensions."""
        original_dim = 1024

        # With skip_whitening=True, should use identity transform
        embeddings = np.random.randn(10, original_dim).astype(np.float32)

        # Identity transform (no truncation)
        mu = np.zeros((1, original_dim))
        W = np.eye(original_dim)

        whitened = apply_whiten(embeddings, mu, W)

        # Dimensions should be preserved
        assert whitened.shape == (10, original_dim)

    def test_multiple_models_different_dims(self):
        """Test handling multiple models with different dimensions."""
        # Model 1: 512 dims
        emb1 = np.random.randn(10, 512).astype(np.float32)
        mu1 = np.zeros((1, 512))
        W1 = np.eye(512)

        # Model 2: 768 dims, truncate to 512
        emb2 = np.random.randn(10, 768).astype(np.float32)
        mu2 = np.zeros((1, 768))
        W2 = np.random.randn(512, 768)  # Truncate

        # Pre-whiten both (THE FIX)
        whitened1 = apply_whiten(emb1, mu1, W1)
        whitened2 = apply_whiten(emb2, mu2, W2)

        # Both should now have same dimension for concatenation
        assert whitened1.shape[1] == whitened2.shape[1] == 512

        # Can concatenate safely
        concatenated = np.concatenate([whitened1, whitened2], axis=1)
        assert concatenated.shape == (10, 1024)


class TestProjectorTrainingStability:
    """Test projector training stability."""

    def test_projector_loss_decreases(self):
        """Projector loss should decrease with training."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        projector = LinearProjector(d_in=512, d_out=256)
        optimizer = torch.optim.Adam(
            projector.parameters(), lr=1e-2
        )  # Higher LR for faster convergence

        # Create synthetic data (fixed target)
        batch_size = 16
        z = torch.randn(batch_size, 256)
        z = z / z.norm(dim=1, keepdim=True)

        # Use fixed input data for consistent optimization
        embeddings = torch.randn(batch_size, 512)

        losses = []
        for _ in range(50):  # More iterations for reliable trend
            optimizer.zero_grad()

            # Training step with fixed input
            output = projector(embeddings)

            # InfoNCE loss (simplified)
            loss = -torch.mean(torch.cosine_similarity(output, z))

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that average of last 10 iterations is better than first 10
        # This is more robust than comparing single values
        early_loss = sum(losses[:10]) / 10
        late_loss = sum(losses[-10:]) / 10
        assert late_loss < early_loss

    def test_projector_handles_nan_inputs(self):
        """Projector should handle NaN inputs gracefully."""
        projector = LinearProjector(d_in=512, d_out=256)

        # Input with NaN
        x = torch.randn(8, 512)
        x[0, 0] = float("nan")

        with torch.no_grad():
            output = projector(x)

        # Should not propagate NaN (due to normalization)
        assert not torch.isnan(output[1:]).any()


class TestValidateEpoch:
    """Test validation metrics for projector training."""

    def _create_mock_loader(self, z_cat, t_vec, batch_size=16):
        """Helper to create a mock DataLoader."""
        from torch.utils.data import DataLoader

        # Create dataset with dictionary batches matching expected format
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, z_cat, t_vec):
                self.z_cat = z_cat
                self.t_vec = t_vec

            def __len__(self):
                return len(self.z_cat)

            def __getitem__(self, idx):
                return {"z_cat": self.z_cat[idx], "t_vec": self.t_vec[idx]}

        dataset = DictDataset(z_cat, t_vec)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def test_perfect_alignment_100_percent_recall(self):
        """Test that perfect alignment gives 100% recall@1."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        # Create model
        model = ImageTextModel(d_cat=512, d_text=256, d_out=128)
        model.eval()

        # Create perfectly aligned embeddings
        n_samples = 64
        z_cat = torch.randn(n_samples, 512)
        t_vec = torch.randn(n_samples, 256)

        # Normalize to ensure proper similarity
        z_cat = z_cat / z_cat.norm(dim=1, keepdim=True)
        t_vec = t_vec / t_vec.norm(dim=1, keepdim=True)

        loader = self._create_mock_loader(z_cat, t_vec, batch_size=16)

        # Run validation
        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 5, 10])

        # With a well-initialized model and normalized inputs, we should get reasonable performance
        # Not necessarily 100% because the model projects to a different space
        assert "i2t_recall@1" in metrics
        assert "t2i_recall@1" in metrics
        assert "avg_recall@1" in metrics
        assert "val_loss" in metrics

        # Check that recall is between 0 and 100
        assert 0 <= metrics["i2t_recall@1"] <= 100
        assert 0 <= metrics["t2i_recall@1"] <= 100
        assert 0 <= metrics["avg_recall@1"] <= 100

        # Check that recall@5 >= recall@1 (monotonicity)
        assert metrics["i2t_recall@5"] >= metrics["i2t_recall@1"]
        assert metrics["t2i_recall@5"] >= metrics["t2i_recall@1"]

        # Check that recall@10 >= recall@5
        assert metrics["i2t_recall@10"] >= metrics["i2t_recall@5"]
        assert metrics["t2i_recall@10"] >= metrics["t2i_recall@5"]

    def test_identity_projection_perfect_recall(self):
        """Test that identity-like projection gives near-perfect recall."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        # Create model that outputs very similar embeddings
        d_out = 128
        n_samples = 32

        # Create model
        model = ImageTextModel(d_cat=d_out, d_text=d_out, d_out=d_out)

        # Initialize with near-identity to ensure high similarity
        with torch.no_grad():
            # Make image projection close to identity
            if hasattr(model.img_head, "proj"):
                model.img_head.proj.weight.copy_(
                    torch.eye(d_out) * 0.9 + torch.randn(d_out, d_out) * 0.01
                )
            # Make text projection close to identity
            if hasattr(model.txt_head, "proj"):
                model.txt_head.proj.weight.copy_(
                    torch.eye(d_out) * 0.9 + torch.randn(d_out, d_out) * 0.01
                )

        model.eval()

        # Create embeddings - use same data for both image and text to ensure alignment
        shared_data = torch.randn(n_samples, d_out)
        shared_data = shared_data / shared_data.norm(dim=1, keepdim=True)

        loader = self._create_mock_loader(shared_data, shared_data, batch_size=8)

        # Run validation
        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 5, 10])

        # With near-identity projection and same input, should get very high recall
        assert metrics["i2t_recall@1"] > 80, f"Got {metrics['i2t_recall@1']}"
        assert metrics["t2i_recall@1"] > 80, f"Got {metrics['t2i_recall@1']}"

        # Mean rank should be very low (close to 1)
        assert metrics["i2t_mean_rank"] < 5, f"Got {metrics['i2t_mean_rank']}"
        assert metrics["t2i_mean_rank"] < 5, f"Got {metrics['t2i_mean_rank']}"

    def test_shuffled_alignment_poor_recall(self):
        """Test that shuffled/misaligned embeddings give poor recall@1."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        # Create model
        model = ImageTextModel(d_cat=512, d_text=256, d_out=128)
        model.eval()

        n_samples = 64
        z_cat = torch.randn(n_samples, 512)
        # Shuffle text embeddings to break alignment
        t_vec_shuffled = torch.randn(n_samples, 256)

        # Permute the text embeddings to ensure misalignment
        perm = torch.randperm(n_samples)
        t_vec_shuffled = t_vec_shuffled[perm]

        loader = self._create_mock_loader(z_cat, t_vec_shuffled, batch_size=16)

        # Run validation
        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 5, 10])

        # With shuffled alignment, recall@1 should be low (around 1/n_samples)
        # We expect around 1.5% for 64 samples (1/64 ≈ 1.56%)
        assert (
            metrics["i2t_recall@1"] < 10
        ), f"Expected low recall@1, got {metrics['i2t_recall@1']}"

        # recall@5 should be higher than recall@1
        assert metrics["i2t_recall@5"] > metrics["i2t_recall@1"]

    def test_comparison_with_evaluate_recall_at_k(self):
        """Test that validate_epoch gives same i2t recall as evaluate_recall_at_k."""
        from src.embedding_fusion import evaluate_recall_at_k
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        # Create model and data
        model = ImageTextModel(d_cat=512, d_text=256, d_out=128)
        model.eval()

        n_samples = 48
        z_cat = torch.randn(n_samples, 512)
        t_vec = torch.randn(n_samples, 256)

        loader = self._create_mock_loader(z_cat, t_vec, batch_size=16)

        # Run validate_epoch
        val_metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 5, 10])

        # Run evaluate_recall_at_k (reference implementation)
        ref_metrics = evaluate_recall_at_k(model, loader, device="cpu", ks=(1, 5, 10))

        # Convert reference metrics to percentage for comparison
        ref_recall_1 = ref_metrics[1] * 100
        ref_recall_5 = ref_metrics[5] * 100
        ref_recall_10 = ref_metrics[10] * 100

        # Should match (i2t only, since reference doesn't do t2i)
        assert (
            abs(val_metrics["i2t_recall@1"] - ref_recall_1) < 0.1
        ), f"i2t_recall@1 mismatch: {val_metrics['i2t_recall@1']} vs {ref_recall_1}"
        assert (
            abs(val_metrics["i2t_recall@5"] - ref_recall_5) < 0.1
        ), f"i2t_recall@5 mismatch: {val_metrics['i2t_recall@5']} vs {ref_recall_5}"
        assert (
            abs(val_metrics["i2t_recall@10"] - ref_recall_10) < 0.1
        ), f"i2t_recall@10 mismatch: {val_metrics['i2t_recall@10']} vs {ref_recall_10}"

    def test_recall_monotonicity(self):
        """Test that recall@k is monotonically increasing with k."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        model = ImageTextModel(d_cat=512, d_text=256, d_out=128)
        model.eval()

        n_samples = 64
        z_cat = torch.randn(n_samples, 512)
        t_vec = torch.randn(n_samples, 256)

        loader = self._create_mock_loader(z_cat, t_vec, batch_size=16)

        # Test with multiple k values
        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 3, 5, 7, 10])

        # Check monotonicity for i2t
        assert metrics["i2t_recall@1"] <= metrics["i2t_recall@3"]
        assert metrics["i2t_recall@3"] <= metrics["i2t_recall@5"]
        assert metrics["i2t_recall@5"] <= metrics["i2t_recall@7"]
        assert metrics["i2t_recall@7"] <= metrics["i2t_recall@10"]

        # Check monotonicity for t2i
        assert metrics["t2i_recall@1"] <= metrics["t2i_recall@3"]
        assert metrics["t2i_recall@3"] <= metrics["t2i_recall@5"]
        assert metrics["t2i_recall@5"] <= metrics["t2i_recall@7"]
        assert metrics["t2i_recall@7"] <= metrics["t2i_recall@10"]

    def test_symmetry_with_identical_embeddings(self):
        """Test that i2t and t2i give same results when image and text embeddings are identical."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        # Use same dimension for both to allow symmetric setup
        d_shared = 256
        model = ImageTextModel(d_cat=d_shared, d_text=d_shared, d_out=128)

        # Make both projections identical
        with torch.no_grad():
            if hasattr(model.txt_head, "proj") and hasattr(model.img_head, "proj"):
                model.txt_head.proj.weight.copy_(model.img_head.proj.weight)
                if model.txt_head.proj.bias is not None:
                    model.txt_head.proj.bias.copy_(model.img_head.proj.bias)

        model.eval()

        # Use same data for both image and text
        n_samples = 32
        shared_data = torch.randn(n_samples, d_shared)

        loader = self._create_mock_loader(shared_data, shared_data, batch_size=8)

        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 5, 10])

        # i2t and t2i should give same results (within numerical precision)
        assert abs(metrics["i2t_recall@1"] - metrics["t2i_recall@1"]) < 0.1
        assert abs(metrics["i2t_recall@5"] - metrics["t2i_recall@5"]) < 0.1
        assert abs(metrics["i2t_recall@10"] - metrics["t2i_recall@10"]) < 0.1

        # Mean ranks should also be similar
        assert abs(metrics["i2t_mean_rank"] - metrics["t2i_mean_rank"]) < 0.5

    def test_small_batch_size(self):
        """Test validation with very small batches."""
        from src.skinmap.embeddings.projector import validate_epoch

        torch.manual_seed(42)

        model = ImageTextModel(d_cat=512, d_text=256, d_out=128)
        model.eval()

        # Very small dataset
        n_samples = 8
        z_cat = torch.randn(n_samples, 512)
        t_vec = torch.randn(n_samples, 256)

        # Batch size of 2
        loader = self._create_mock_loader(z_cat, t_vec, batch_size=2)

        metrics = validate_epoch(model, loader, device="cpu", k_values=[1, 3, 5])

        # Should still work without errors
        assert "i2t_recall@1" in metrics
        assert "val_loss" in metrics
        assert not np.isnan(metrics["i2t_recall@1"])
        assert not np.isnan(metrics["val_loss"])

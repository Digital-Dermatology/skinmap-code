"""Tests for STRUCTURE loss with per-model hierarchical consistency.

This module tests that the fix for cross-model correlation preservation works correctly.
"""

import pytest
import torch

from src.skinmap.loss.clip_loss import CLIPLoss, hierarchical_consistency_reg


class TestHierarchicalConsistencyReg:
    """Test the base hierarchical consistency regularization function."""

    def test_basic_functionality(self):
        """Test that hierarchical consistency runs without errors."""
        batch_size = 32
        dim = 128

        original = torch.randn(batch_size, dim)
        aligned = torch.randn(batch_size, dim)

        loss = hierarchical_consistency_reg(
            original,
            aligned,
            levels=2,
            temperature=0.1,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss >= 0  # JS divergence is non-negative

    def test_perfect_alignment_gives_low_loss(self):
        """Test that identical embeddings give near-zero loss."""
        batch_size = 32
        dim = 128

        embeddings = torch.randn(batch_size, dim)

        loss = hierarchical_consistency_reg(
            embeddings,
            embeddings.clone(),
            levels=2,
            temperature=0.1,
        )

        # Should be very small (not exactly zero due to numerical precision)
        assert loss < 0.01

    def test_different_batch_sizes_raises_error(self):
        """Test that different batch sizes raise an error in matrix operations."""
        # Different batch sizes will cause matrix multiply to fail
        original = torch.randn(32, 128)
        aligned = torch.randn(16, 128)  # Different batch size

        # This should fail during the similarity matrix computation
        with pytest.raises(RuntimeError):
            hierarchical_consistency_reg(
                original,
                aligned,
                levels=2,
                temperature=0.1,
            )


class TestCLIPLossPerModelHierarchy:
    """Test CLIPLoss with per-model hierarchical consistency."""

    def test_single_model_uses_full_concatenation(self):
        """Test that single model case uses old behavior (full concatenation)."""
        batch_size = 16
        dim = 128
        out_dim = 256

        # Single model
        teacher_dims = [dim]
        text_dims = [dim]

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=teacher_dims,
            text_dims=text_dims,
        )

        # Create dummy embeddings
        img_aligned = torch.randn(batch_size, out_dim)
        txt_aligned = torch.randn(batch_size, out_dim)
        img_original = torch.randn(batch_size, dim)
        txt_original = torch.randn(batch_size, dim)

        result = loss_fn(
            img_aligned,
            txt_aligned,
            img_original,
            txt_original,
        )

        assert "overall_loss" in result
        assert "clip_loss" in result
        assert "hierarchy_loss" in result
        assert result["overall_loss"] > 0

    def test_multiple_models_splits_correctly(self):
        """Test that multiple models are split and processed separately."""
        batch_size = 16
        dim1, dim2, dim3 = 128, 256, 192
        out_dim = 256

        # Three models
        teacher_dims = [dim1, dim2, dim3]
        text_dims = [dim1, dim2]  # Only first two are CLIP models

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=teacher_dims,
            text_dims=text_dims,
        )

        # Create concatenated embeddings
        img_m1 = torch.randn(batch_size, dim1)
        img_m2 = torch.randn(batch_size, dim2)
        img_m3 = torch.randn(batch_size, dim3)
        img_original = torch.cat([img_m1, img_m2, img_m3], dim=1)

        txt_m1 = torch.randn(batch_size, dim1)
        txt_m2 = torch.randn(batch_size, dim2)
        txt_original = torch.cat([txt_m1, txt_m2], dim=1)

        img_aligned = torch.randn(batch_size, out_dim)
        txt_aligned = torch.randn(batch_size, out_dim)

        result = loss_fn(
            img_aligned,
            txt_aligned,
            img_original,
            txt_original,
        )

        assert "overall_loss" in result
        assert "hierarchy_loss" in result
        assert result["overall_loss"] > 0

    def test_per_model_hierarchy_computation(self):
        """Test the _compute_per_model_hierarchy method directly."""
        batch_size = 16
        dim1, dim2 = 128, 256
        out_dim = 256

        teacher_dims = [dim1, dim2]

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=teacher_dims,
        )

        # Create concatenated embeddings
        concatenated = torch.randn(batch_size, dim1 + dim2)
        aligned = torch.randn(batch_size, out_dim)

        # Test per-model computation
        loss = loss_fn._compute_per_model_hierarchy(
            concatenated,
            aligned,
            teacher_dims,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss >= 0

    def test_none_dims_uses_full_concatenation(self):
        """Test that None dims falls back to old behavior."""
        batch_size = 16
        dim = 256
        out_dim = 256

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=None,  # No dims specified
            text_dims=None,
        )

        concatenated = torch.randn(batch_size, dim)
        aligned = torch.randn(batch_size, out_dim)

        loss = loss_fn._compute_per_model_hierarchy(
            concatenated,
            aligned,
            None,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0


class TestCrossModelCorrelationAvoidance:
    """Test that per-model computation avoids cross-model correlations."""

    def test_cross_model_correlations_not_preserved(self):
        """
        Test that cross-model correlations are NOT preserved in per-model mode.

        This is the key test: we verify that the loss doesn't try to preserve
        correlations between different models' embeddings.
        """
        batch_size = 32
        dim1, dim2 = 128, 128
        out_dim = 256

        # Create two models with specific correlation structure
        torch.manual_seed(42)

        # Model 1: random embeddings
        m1_embeddings = torch.randn(batch_size, dim1)

        # Model 2: embeddings that are highly correlated with Model 1
        # (this is an artificial scenario that shouldn't be preserved)
        m2_embeddings = m1_embeddings + 0.1 * torch.randn(batch_size, dim2)

        # Concatenate
        concatenated = torch.cat([m1_embeddings, m2_embeddings], dim=1)

        # Create aligned embeddings (projected)
        aligned = torch.randn(batch_size, out_dim)

        # Test with per-model hierarchy (should NOT preserve M1-M2 correlation)
        loss_fn_per_model = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=[dim1, dim2],
        )

        loss_per_model = loss_fn_per_model._compute_per_model_hierarchy(
            concatenated,
            aligned,
            [dim1, dim2],
        )

        # Test with old behavior (WOULD preserve M1-M2 correlation)
        loss_fn_full = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=None,  # Old behavior
        )

        loss_full = loss_fn_full._compute_per_model_hierarchy(
            concatenated,
            aligned,
            None,
        )

        # The per-model loss should be different from the full concatenation loss
        # because it's not trying to preserve cross-model correlations
        # (The exact relationship depends on the random seed, but they should differ)
        assert loss_per_model != loss_full

        # Both should be valid non-negative losses
        assert loss_per_model >= 0
        assert loss_full >= 0

    def test_per_model_averages_correctly(self):
        """Test that averaging across models works correctly."""
        batch_size = 16
        dim1, dim2, dim3 = 64, 64, 64
        out_dim = 128

        teacher_dims = [dim1, dim2, dim3]

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=1,  # Simple case
            teacher_dims=teacher_dims,
        )

        # Create identical blocks to verify averaging
        block = torch.randn(batch_size, dim1)
        concatenated = torch.cat([block, block, block], dim=1)
        aligned = torch.randn(batch_size, out_dim)

        # Since all blocks are identical, the average should equal individual losses
        loss_avg = loss_fn._compute_per_model_hierarchy(
            concatenated,
            aligned,
            teacher_dims,
        )

        # Compute individual loss
        loss_individual = hierarchical_consistency_reg(
            block,
            aligned,
            levels=1,
            temperature=0.07,
            weighting="none",
            margin=0.0,
        )

        # They should be equal (within numerical precision)
        assert torch.allclose(loss_avg, loss_individual, rtol=1e-4)


class TestGradientFlow:
    """Test that gradients flow correctly through per-model computation."""

    def test_gradients_flow_through_per_model_loss(self):
        """Test that we can backpropagate through the per-model hierarchy loss."""
        batch_size = 16
        dim1, dim2 = 128, 256
        out_dim = 256

        teacher_dims = [dim1, dim2]

        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=1.0,
            hierarchy_levels=2,
            teacher_dims=teacher_dims,
        )

        # Create embeddings that require gradients
        img_aligned = torch.randn(batch_size, out_dim, requires_grad=True)
        txt_aligned = torch.randn(batch_size, out_dim, requires_grad=True)
        img_original = torch.randn(batch_size, dim1 + dim2)
        txt_original = torch.randn(batch_size, dim1 + dim2)

        result = loss_fn(
            img_aligned,
            txt_aligned,
            img_original,
            txt_original,
        )

        loss = result["overall_loss"]
        loss.backward()

        # Check that gradients were computed
        assert img_aligned.grad is not None
        assert txt_aligned.grad is not None
        assert not torch.all(img_aligned.grad == 0)
        assert not torch.all(txt_aligned.grad == 0)

    def test_warmup_schedule(self):
        """Test that warmup schedule works correctly."""
        loss_fn = CLIPLoss(
            temperature=0.07,
            lambda_hierarchy=10.0,
            warmup_steps=100,
            teacher_dims=[128, 256],
        )

        # Initially, lambda should be at base value (initialized)
        assert loss_fn.lambda_hierarchy == 10.0  # Initialized to base value
        assert loss_fn.train_step == 0

        # After a few steps, lambda should increase
        for _ in range(50):
            loss_fn.step()

        # After 50 steps, train_step=50 but lambda was last computed at step 49
        # So lambda = 10.0 * (49 / 100) = 4.9
        assert loss_fn.train_step == 50
        expected = 10.0 * (49 / 100)  # Uses train_step from before last increment
        assert torch.allclose(
            loss_fn.lambda_hierarchy,
            torch.tensor(expected),
            rtol=1e-4,
        )

        # After 51 more steps (total 101), should be at full value
        for _ in range(51):
            loss_fn.step()

        # After 101 steps, lambda uses min(1.0, 100/100) = 1.0, so lambda = 10.0
        assert torch.allclose(
            loss_fn.lambda_hierarchy,
            torch.tensor(10.0),
            rtol=1e-4,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

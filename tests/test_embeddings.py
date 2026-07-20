"""Tests for embedding extraction functionality.

Critical tests for bug fixes:
- Corrupted sample handling with negative index marking
- DataFrame-embedding synchronization
- Edge cases with idx=0 corrupted samples
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.skinmap.embeddings.extractors import (
    ImageDataset,
    _filter_corrupted,
    extract_clip_embeddings,
    extract_ssl_embeddings,
)


class TestImageDataset:
    """Test unified ImageDataset class."""

    def test_valid_image_returns_positive_index(self, sample_df):
        """Valid images should return positive indices."""
        dataset = ImageDataset(sample_df, transform=None, text_col="description")

        img, text, idx = dataset[0]
        assert idx == 0  # Positive index
        assert isinstance(img, Image.Image)
        assert text == "Image 0"

    def test_corrupted_image_returns_negative_index(self, temp_dir):
        """Corrupted images should return -idx-1."""
        # Create df with non-existent path
        df = pd.DataFrame(
            {
                "img_path": ["/nonexistent/image.jpg"],
                "description": ["Test"],
            }
        )

        dataset = ImageDataset(df, transform=None, text_col="description")
        img, text, idx = dataset[0]

        # Should return dummy image and negative index
        assert idx == -1  # -0-1 = -1
        assert isinstance(img, Image.Image)
        assert text == "unknown"

    def test_corrupted_at_idx_zero_handled_correctly(self, temp_dir):
        """Critical edge case: idx=0 corrupted should be -1, not -0."""
        df = pd.DataFrame(
            {
                "img_path": ["/nonexistent/img_0.jpg"],
                "description": ["Test"],
            }
        )

        dataset = ImageDataset(df)
        _, _, idx = dataset[0]

        # idx=0 corrupted should be -0-1 = -1 (not -0 which equals 0!)
        assert idx == -1
        assert idx < 0  # Must be negative to mark as corrupted

    def test_multiple_corrupted_samples(self, temp_dir):
        """Test multiple corrupted samples with correct indexing."""
        # Create mix of valid and corrupted
        valid_path = temp_dir + "/valid.jpg"
        Image.new("RGB", (224, 224)).save(valid_path)

        df = pd.DataFrame(
            {
                "img_path": [
                    valid_path,  # idx=0, valid
                    "/corrupt1.jpg",  # idx=1, corrupted → -1-1 = -2
                    "/corrupt2.jpg",  # idx=2, corrupted → -2-1 = -3
                    valid_path,  # idx=3, valid
                ],
                "description": ["Test"] * 4,
            }
        )

        dataset = ImageDataset(df)

        _, _, idx0 = dataset[0]
        _, _, idx1 = dataset[1]
        _, _, idx2 = dataset[2]
        _, _, idx3 = dataset[3]

        assert idx0 == 0  # Valid
        assert idx1 == -2  # Corrupted: -1-1
        assert idx2 == -3  # Corrupted: -2-1
        assert idx3 == 3  # Valid

    def test_dataset_without_text_column(self, sample_df):
        """Test dataset without text column (SSL case)."""
        dataset = ImageDataset(sample_df, transform=None, text_col=None)

        img, text, idx = dataset[0]
        assert text is None  # No text for SSL
        assert idx == 0

    def test_dataset_with_transforms(self, sample_df):
        """Test dataset with image transforms."""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        dataset = ImageDataset(sample_df, transform=transform, text_col=None)
        img, text, idx = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)


class TestFilterCorrupted:
    """Test corrupted sample filtering function."""

    def test_filter_removes_corrupted_samples(self):
        """Filter should remove samples with negative indices."""
        embeddings = np.random.randn(5, 512)
        indices = np.array([0, -2, 2, -4, 4])  # indices 1 and 3 are corrupted
        df = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(5)]})

        filtered_embs, _, corrupted_list, filtered_df = _filter_corrupted(
            embeddings, indices, df
        )

        assert len(filtered_embs) == 3  # Only valid samples
        assert len(filtered_df) == 3
        assert len(corrupted_list) == 2
        assert 1 in corrupted_list  # Original idx 1 (marked as -2)
        assert 3 in corrupted_list  # Original idx 3 (marked as -4)

    def test_filter_decodes_negative_indices_correctly(self):
        """Filter should correctly decode -idx-1 back to original idx."""
        embeddings = np.random.randn(3, 512)
        indices = np.array([-1, -2, -3])  # All corrupted: idx 0, 1, 2

        df = pd.DataFrame({"img_path": ["a.jpg", "b.jpg", "c.jpg"]})

        _, _, corrupted_list, _ = _filter_corrupted(embeddings, indices, df)

        # Should decode: -1 → 0, -2 → 1, -3 → 2
        assert corrupted_list == [0, 1, 2]

    def test_filter_preserves_order_of_valid_samples(self):
        """Filter should preserve order of valid samples."""
        embeddings = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        indices = np.array([0, -2, 2, -4, 4])
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})

        filtered_embs, _, _, filtered_df = _filter_corrupted(embeddings, indices, df)

        # Should have samples at original indices 0, 2, 4
        np.testing.assert_array_equal(filtered_embs, np.array([[1, 1], [3, 3], [5, 5]]))
        assert list(filtered_df["value"]) == [10, 30, 50]

    def test_filter_with_text_embeddings(self):
        """Filter should handle text embeddings correctly."""
        image_embs = np.random.randn(3, 512)
        text_embs = np.random.randn(3, 512)
        indices = np.array([0, -2, 2])
        df = pd.DataFrame({"img_path": ["a.jpg", "b.jpg", "c.jpg"]})

        filtered_image, filtered_text, corrupted_list, filtered_df = _filter_corrupted(
            image_embs, indices, df, text_embeddings=text_embs
        )

        assert len(filtered_image) == 2
        assert len(filtered_text) == 2
        assert corrupted_list == [1]

    def test_filter_all_valid_samples(self):
        """Filter with all valid samples should keep everything."""
        embeddings = np.random.randn(5, 512)
        indices = np.array([0, 1, 2, 3, 4])  # All valid
        df = pd.DataFrame({"img_path": [f"img_{i}.jpg" for i in range(5)]})

        filtered_embs, _, corrupted_list, filtered_df = _filter_corrupted(
            embeddings, indices, df
        )

        assert len(filtered_embs) == 5
        assert len(filtered_df) == 5
        assert len(corrupted_list) == 0

    def test_filter_all_corrupted_samples(self):
        """Filter with all corrupted samples should return empty."""
        embeddings = np.random.randn(3, 512)
        indices = np.array([-1, -2, -3])  # All corrupted
        df = pd.DataFrame({"img_path": ["a.jpg", "b.jpg", "c.jpg"]})

        filtered_embs, _, corrupted_list, filtered_df = _filter_corrupted(
            embeddings, indices, df
        )

        assert len(filtered_embs) == 0
        assert len(filtered_df) == 0
        assert len(corrupted_list) == 3

    def test_dataframe_embedding_sync_maintained(self):
        """Critical: DataFrame and embeddings must stay synchronized."""
        embeddings = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        indices = np.array([0, -2, 2, -4, 4])  # Remove idx 1 and 3
        df = pd.DataFrame(
            {
                "img_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
                "label": ["A", "B", "C", "D", "E"],
            }
        )

        filtered_embs, _, _, filtered_df = _filter_corrupted(embeddings, indices, df)

        # Check sync: embedding value should match label
        assert filtered_embs[0, 0] == 1 and filtered_df.iloc[0]["label"] == "A"
        assert filtered_embs[1, 0] == 3 and filtered_df.iloc[1]["label"] == "C"
        assert filtered_embs[2, 0] == 5 and filtered_df.iloc[2]["label"] == "E"


class TestExtractCLIPEmbeddings:
    """Test CLIP embedding extraction."""

    def test_extract_returns_correct_shapes(
        self, mock_clip_model, mock_processor, sample_df, device
    ):
        """Extract should return correct embedding shapes."""
        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            sample_df,
            device,
            batch_size=4,
            dataset_col="dataset_desc",
        )

        assert image_embs.shape[0] == len(sample_df)
        assert image_embs.shape[1] == 512  # Mock model output dim
        assert text_embs.shape == image_embs.shape
        assert len(labels) == len(sample_df)
        assert len(filtered_df) == len(sample_df)

    def test_extract_handles_corrupted_samples(
        self, mock_clip_model, mock_processor, sample_df_with_corrupted, device
    ):
        """Extract should handle corrupted samples correctly."""
        df, num_corrupted = sample_df_with_corrupted

        image_embs, text_embs, labels, corrupted_list, filtered_df = (
            extract_clip_embeddings(
                mock_clip_model,
                mock_processor,
                df,
                device,
                batch_size=4,
                dataset_col="dataset_desc",
            )
        )

        # Should filter out corrupted samples
        assert len(filtered_df) == len(df) - num_corrupted
        assert len(image_embs) == len(filtered_df)
        assert len(corrupted_list) == num_corrupted

    def test_extract_with_max_samples(
        self, mock_clip_model, mock_processor, sample_df, device
    ):
        """Extract should respect max_samples limit."""
        image_embs, _, _, _, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            sample_df,
            device,
            batch_size=4,
            max_samples=5,
        )

        assert len(image_embs) <= 5
        assert len(filtered_df) <= 5


class TestExtractSSLEmbeddings:
    """Test SSL embedding extraction."""

    def test_ssl_extract_returns_correct_shapes(
        self, mock_ssl_model, sample_df, device
    ):
        """SSL extract should return correct shapes."""
        image_embs, text_embs, labels, corrupted, filtered_df = extract_ssl_embeddings(
            mock_ssl_model,
            sample_df,
            device,
            batch_size=4,
            dataset_col="dataset_desc",
        )

        assert image_embs.shape[0] == len(sample_df)
        assert image_embs.shape[1] == 768  # Mock SSL model output
        assert text_embs is None  # SSL models don't produce text embeddings
        assert len(labels) == len(sample_df)

    def test_ssl_handles_corrupted_samples(
        self, mock_ssl_model, sample_df_with_corrupted, device
    ):
        """SSL extraction should handle corrupted samples."""
        df, num_corrupted = sample_df_with_corrupted

        image_embs, _, _, corrupted_list, filtered_df = extract_ssl_embeddings(
            mock_ssl_model,
            df,
            device,
            batch_size=4,
        )

        assert len(filtered_df) == len(df) - num_corrupted
        assert len(corrupted_list) == num_corrupted


class TestEdgeCases:
    """Test edge cases in embedding extraction."""

    def test_empty_dataframe(self, mock_clip_model, mock_processor, device):
        """Test extraction with empty dataframe."""
        df = pd.DataFrame(columns=["img_path", "description", "dataset_desc"])

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df,
            device,
            batch_size=4,
        )

        assert len(image_embs) == 0
        assert len(filtered_df) == 0

    def test_single_sample(self, mock_clip_model, mock_processor, sample_df, device):
        """Test extraction with single sample."""
        df_single = sample_df.iloc[:1]

        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            df_single,
            device,
            batch_size=4,
        )

        assert len(image_embs) == 1
        assert len(filtered_df) == 1

    def test_batch_size_larger_than_dataset(
        self, mock_clip_model, mock_processor, sample_df, device
    ):
        """Test with batch size larger than dataset."""
        image_embs, text_embs, labels, corrupted, filtered_df = extract_clip_embeddings(
            mock_clip_model,
            mock_processor,
            sample_df,
            device,
            batch_size=1000,  # Larger than dataset
        )

        assert len(image_embs) == len(sample_df)
        assert len(filtered_df) == len(sample_df)

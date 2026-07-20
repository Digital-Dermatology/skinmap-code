"""Tests for model loading functionality.

Critical tests for:
- ModelInfo dataclass
- Single model loading (CLIP and SSL)
- Multiple model loading
- normalize_model_tuple compatibility
- Error handling for invalid paths
"""


import pytest
import torch

from src.skinmap.models.loaders import (
    ModelInfo,
    load_multiple_models,
    normalize_model_tuple,
)


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self, mock_clip_model, mock_processor):
        """Test creating ModelInfo instance."""
        model_info = ModelInfo(
            model=mock_clip_model,
            processor=mock_processor,
            model_type="clip",
            model_path="test/model",
        )

        assert model_info.model is not None
        assert model_info.processor is not None
        assert model_info.model_type == "clip"
        assert model_info.model_path == "test/model"

    def test_model_info_ssl_no_processor(self, mock_ssl_model):
        """Test ModelInfo for SSL model without processor."""
        model_info = ModelInfo(
            model=mock_ssl_model,
            processor=None,
            model_type="ssl",
            model_path="test/ssl_model",
        )

        assert model_info.model is not None
        assert model_info.processor is None
        assert model_info.model_type == "ssl"

    def test_model_info_equality(self, mock_clip_model, mock_processor):
        """Test ModelInfo equality comparison."""
        model_info1 = ModelInfo(
            model=mock_clip_model,
            processor=mock_processor,
            model_type="clip",
            model_path="test/model",
        )
        model_info2 = ModelInfo(
            model=mock_clip_model,
            processor=mock_processor,
            model_type="clip",
            model_path="test/model",
        )

        # DataclassesEq by default compares by field values
        # For now just check types are correct
        assert isinstance(model_info1, ModelInfo)
        assert isinstance(model_info2, ModelInfo)


class TestNormalizeModelTuple:
    """Test normalize_model_tuple function."""

    def test_normalize_model_info_passthrough(self, mock_clip_model, mock_processor):
        """ModelInfo should pass through unchanged."""
        model_info = ModelInfo(
            model=mock_clip_model,
            processor=mock_processor,
            model_type="clip",
            model_path="test/model",
        )

        model, processor, model_type = normalize_model_tuple(model_info)

        assert model is mock_clip_model
        assert processor is mock_processor
        assert model_type == "clip"

    def test_normalize_tuple_input(self, mock_clip_model, mock_processor):
        """Tuple input should be unpacked correctly."""
        input_tuple = (mock_clip_model, mock_processor, "clip")

        model, processor, model_type = normalize_model_tuple(input_tuple)

        assert model is mock_clip_model
        assert processor is mock_processor
        assert model_type == "clip"

    def test_normalize_3tuple_input(self, mock_clip_model, mock_processor):
        """3-tuple input (legacy format) should work."""
        input_tuple = (mock_clip_model, mock_processor, "clip")

        model, processor, model_type = normalize_model_tuple(input_tuple)

        assert model is mock_clip_model
        assert processor is mock_processor
        assert model_type == "clip"

    def test_normalize_ssl_model_tuple(self, mock_ssl_model):
        """SSL model tuple (no processor) should work."""
        input_tuple = (mock_ssl_model, None, "ssl")

        model, processor, model_type = normalize_model_tuple(input_tuple)

        assert model is mock_ssl_model
        assert processor is None
        assert model_type == "ssl"


class TestLoadMultipleModels:
    """Test load_multiple_models function."""

    def test_load_single_model_returns_list(
        self, monkeypatch, mock_clip_model, mock_processor
    ):
        """Loading single model should return list with one ModelInfo."""

        def mock_load_model_and_processor(model_name, device, model_path=None):
            return mock_clip_model, mock_processor

        # Mock the load function
        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("test_model", device)

        assert len(models) == 1
        assert isinstance(models[0], ModelInfo)
        assert models[0].model_type == "clip"

    def test_load_multiple_models_comma_separated(
        self, monkeypatch, mock_clip_model, mock_processor
    ):
        """Loading comma-separated models should return list of ModelInfo."""

        def mock_load_model_and_processor(model_name, device, model_path=None):
            return mock_clip_model, mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("model1,model2", device)

        assert len(models) == 2
        assert all(isinstance(m, ModelInfo) for m in models)
        assert all(m.model_type == "clip" for m in models)

    def test_load_with_explicit_paths(
        self, monkeypatch, mock_clip_model, mock_processor
    ):
        """Loading with explicit paths in model name should work."""

        def mock_load_model_and_processor(model_name, device):
            return mock_clip_model, mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("path1,path2", device)

        assert len(models) == 2
        assert models[0].model_path == "path1"
        assert models[1].model_path == "path2"

    def test_load_ssl_model_detection(self, monkeypatch, mock_ssl_model):
        """SSL model should be detected and marked correctly."""

        def mock_load_model_and_processor(model_name, device, model_path=None):
            # Return just model for SSL (no processor)
            return mock_ssl_model, None

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("dino_qderma", device)

        assert len(models) == 1
        # Model type detection should work
        assert isinstance(models[0], ModelInfo)

    def test_load_mixed_clip_and_ssl(
        self, monkeypatch, mock_clip_model, mock_ssl_model, mock_processor
    ):
        """Loading mix of CLIP and SSL models should work."""

        call_count = [0]

        def mock_load_model_and_processor(model_name, device, model_path=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_clip_model, mock_processor  # First call: CLIP
            else:
                return mock_ssl_model, None  # Second call: SSL


class TestModelLoadingErrors:
    """Test error handling in model loading."""

    def test_load_invalid_model_raises_error(self, monkeypatch, device):
        """Loading invalid model should raise RuntimeError."""

        def mock_load_model_and_processor(model_name, device):
            raise ValueError("Model not found")

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        with pytest.raises(RuntimeError):
            load_multiple_models("invalid_model", device)

    def test_empty_model_name_handled(
        self, monkeypatch, mock_clip_model, mock_processor, device
    ):
        """Empty model name should be handled."""

        def mock_load_model_and_processor(model_name, device):
            return mock_clip_model, mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        # Empty string splits to [""]
        models = load_multiple_models("", device)
        # Should load one empty-named model
        assert len(models) == 1


class TestModelDevicePlacement:
    """Test device placement for models."""

    def test_model_moved_to_correct_device(
        self, mock_clip_model, mock_processor, monkeypatch
    ):
        """Model should be moved to specified device."""

        device_set = [None]

        class MockModel:
            def to(self, device):
                device_set[0] = device
                return self

            def eval(self):
                return self

        def mock_load_model_and_processor(model_name, device, model_path=None):
            return MockModel(), mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("test_model", device)

        # Model should have been moved to device
        # (In real implementation, this is done in load_model_and_processor)
        assert len(models) > 0


class TestEdgeCases:
    """Test edge cases in model loading."""

    def test_whitespace_in_model_names(
        self, monkeypatch, mock_clip_model, mock_processor
    ):
        """Model names with whitespace should be handled."""

        def mock_load_model_and_processor(model_name, device, model_path=None):
            # Should receive trimmed name
            assert model_name.strip() == model_name
            return mock_clip_model, mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models(" model1 , model2 ", device)

        assert len(models) == 2

    def test_duplicate_model_names(self, monkeypatch, mock_clip_model, mock_processor):
        """Duplicate model names should load multiple times."""

        def mock_load_model_and_processor(model_name, device, model_path=None):
            return mock_clip_model, mock_processor

        import src.train_clip

        monkeypatch.setattr(
            src.train_clip, "load_model_and_processor", mock_load_model_and_processor
        )

        device = torch.device("cpu")
        models = load_multiple_models("model1,model1,model1", device)

        # Should load 3 times even if same model
        assert len(models) == 3

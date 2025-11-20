"""Tests for enterprise features."""

import pytest
import torch

from cosmos_tokenizer.core.modality import (
    Modality,
    ModalityAwareTokenizer,
    TokenizerRegistry,
)
from cosmos_tokenizer.enterprise import (
    BillingEngine,
    CachedTokenizer,
    CostCalculator,
    DynamicBatcher,
    EnterpriseImageTokenizer,
    PricingTier,
    SafetyValidator,
    TokenCounter,
    TokenizerMetadata,
    UsageTracker,
    VersionManager,
)


class TestModalityAwareTokenizer:
    """Test modality-aware tokenization architecture."""

    def test_registry_registration(self):
        """Test tokenizer registry."""
        registry = TokenizerRegistry()

        # Create a mock tokenizer
        tokenizer = EnterpriseImageTokenizer(mode="DI", spatial_compression=16)

        # Register
        registry.register(Modality.IMAGE, tokenizer)

        # Retrieve
        retrieved = registry.get(Modality.IMAGE)
        assert retrieved is tokenizer

        # List modalities
        modalities = registry.list_modalities()
        assert Modality.IMAGE in modalities

    def test_registry_duplicate_registration(self):
        """Test that duplicate registration raises error."""
        registry = TokenizerRegistry()
        tokenizer = EnterpriseImageTokenizer(mode="DI", spatial_compression=16)

        registry.register(Modality.IMAGE, tokenizer)

        # Should raise ValueError
        with pytest.raises(ValueError):
            registry.register(Modality.IMAGE, tokenizer)

    def test_registry_missing_modality(self):
        """Test that missing modality raises error."""
        registry = TokenizerRegistry()

        with pytest.raises(KeyError):
            registry.get(Modality.AUDIO)


class TestTokenCounter:
    """Test token counting functionality."""

    def test_count_image_tokens(self):
        """Test image token counting."""
        counter = TokenCounter()

        image = torch.randn(1, 3, 256, 256)
        token_count = counter.count_image_tokens(
            image, spatial_compression=16, mode="DI"
        )

        # For DI mode with 256x256 image and compression 16:
        # (256/16) * (256/16) = 16 * 16 = 256 tokens
        assert token_count == 256

    def test_count_video_tokens(self):
        """Test video token counting."""
        counter = TokenCounter()

        video = torch.randn(1, 3, 16, 256, 256)
        token_count = counter.count_video_tokens(
            video,
            spatial_compression=16,
            temporal_compression=8,
            mode="DV",
        )

        # For DV mode: (16/8) * (256/16) * (256/16) = 2 * 16 * 16 = 512
        assert token_count == 512

    def test_record_usage(self):
        """Test usage recording."""
        counter = TokenCounter()

        usage = counter.record_usage(
            modality=Modality.IMAGE,
            token_count=256,
            operation="encode",
            user_id="user-123",
        )

        assert usage.modality == Modality.IMAGE
        assert usage.token_count == 256
        assert usage.user_id == "user-123"

        # Check totals
        assert counter.get_total_tokens() == 256

    def test_usage_summary(self):
        """Test usage summary."""
        counter = TokenCounter()

        counter.record_usage(Modality.IMAGE, 256, "encode", "user-1")
        counter.record_usage(Modality.VIDEO, 512, "encode", "user-2")

        summary = counter.get_summary()

        assert summary["total_tokens"] == 768
        assert summary["by_modality"]["image"] == 256
        assert summary["by_modality"]["video"] == 512


class TestBillingEngine:
    """Test billing functionality."""

    def test_cost_calculation(self):
        """Test cost calculation."""
        calculator = CostCalculator(tier=PricingTier.PRO)

        # Image encoding: 1M tokens at $0.10/M = $0.10
        cost = calculator.calculate_cost(
            modality=Modality.IMAGE,
            token_count=1_000_000,
            operation="encode",
        )

        assert cost == pytest.approx(0.10, rel=0.01)

    def test_free_tier_pricing(self):
        """Test free tier pricing."""
        calculator = CostCalculator(tier=PricingTier.FREE)

        cost = calculator.calculate_cost(
            modality=Modality.IMAGE,
            token_count=1_000_000,
            operation="encode",
        )

        assert cost == 0.0

    def test_quota_check(self):
        """Test quota enforcement."""
        calculator = CostCalculator(tier=PricingTier.FREE)

        # Within quota
        within, remaining, msg = calculator.check_quota(500_000)
        assert within is True
        assert remaining == 500_000

        # Over quota
        within, remaining, msg = calculator.check_quota(2_000_000)
        assert within is False
        assert msg is not None

    def test_billing_engine(self):
        """Test billing engine."""
        engine = BillingEngine(tier=PricingTier.PRO)

        # Record usage
        engine.record_usage(
            modality=Modality.IMAGE,
            token_count=1_000_000,
            operation="encode",
            user_id="user-123",
        )

        # Generate bill
        bill = engine.generate_bill(user_id="user-123")

        assert bill["total_tokens"] == 1_000_000
        assert bill["total_cost"] == pytest.approx(0.10, rel=0.01)


class TestSafetyValidator:
    """Test safety validation."""

    def test_valid_image(self):
        """Test validation of valid image."""
        validator = SafetyValidator()

        image = torch.randn(1, 3, 256, 256) * 0.5  # Range [-0.5, 0.5]
        result = validator.validate_image(image)

        assert result.is_valid is True
        assert result.error_message is None

    def test_invalid_shape(self):
        """Test validation of invalid shape."""
        validator = SafetyValidator()

        # Wrong number of channels
        image = torch.randn(1, 4, 256, 256)
        result = validator.validate_image(image)

        assert result.is_valid is False
        assert "3 channels" in result.error_message

    def test_resolution_limit(self):
        """Test resolution limit enforcement."""
        validator = SafetyValidator(max_image_resolution=512)

        # Exceeds limit
        image = torch.randn(1, 3, 1024, 1024)
        result = validator.validate_image(image)

        assert result.is_valid is False
        assert "exceeds" in result.error_message.lower()

    def test_rate_limiting(self):
        """Test rate limiting."""
        validator = SafetyValidator(rate_limit_requests_per_minute=2)

        # First two requests should succeed
        can_proceed, wait_time = validator.check_rate_limit("user-1")
        assert can_proceed is True

        can_proceed, wait_time = validator.check_rate_limit("user-1")
        assert can_proceed is True

        # Third request should be rate limited
        can_proceed, wait_time = validator.check_rate_limit("user-1")
        assert can_proceed is False
        assert wait_time > 0


class TestUsageTracker:
    """Test usage tracking."""

    def test_record_usage(self):
        """Test usage recording."""
        tracker = UsageTracker()

        record = tracker.record(
            modality=Modality.IMAGE,
            operation="encode",
            token_count=256,
            latency_ms=10.5,
            throughput_tokens_per_sec=24380.0,
            user_id="user-1",
        )

        assert record.modality == Modality.IMAGE
        assert record.token_count == 256

    def test_get_metrics(self):
        """Test metrics calculation."""
        tracker = UsageTracker()

        # Record multiple operations
        for i in range(10):
            tracker.record(
                modality=Modality.IMAGE,
                operation="encode",
                token_count=256,
                latency_ms=10.0 + i,
                throughput_tokens_per_sec=25000.0,
            )

        metrics = tracker.get_metrics()

        assert metrics["count"] == 10
        assert metrics["total_tokens"] == 2560
        assert metrics["avg_latency_ms"] == pytest.approx(14.5, rel=0.1)


class TestVersioning:
    """Test versioning system."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        metadata = TokenizerMetadata(
            version="1.0.0",
            modality="image",
            mode="DI",
            spatial_compression=16,
            codebook_size=65536,
        )

        assert metadata.version == "1.0.0"
        assert metadata.modality == "image"

    def test_metadata_serialization(self):
        """Test metadata serialization."""
        metadata = TokenizerMetadata(
            version="1.0.0",
            modality="image",
            mode="DI",
            spatial_compression=16,
        )

        # Convert to dict and back
        data = metadata.to_dict()
        restored = TokenizerMetadata.from_dict(data)

        assert restored.version == metadata.version
        assert restored.modality == metadata.modality


class TestBatchProcessing:
    """Test batch processing utilities."""

    def test_dynamic_batcher(self):
        """Test dynamic batching."""
        batcher = DynamicBatcher(max_batch_size=8)

        # Create sample inputs
        inputs = [torch.randn(1, 3, 256, 256) for _ in range(20)]

        # Batch inputs
        batches = batcher.batch_inputs(inputs, batch_size=4)

        assert len(batches) == 5  # 20 / 4 = 5 batches
        assert batches[0].shape[0] == 4


class TestEnterpriseImageTokenizer:
    """Test enterprise image tokenizer wrapper."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = EnterpriseImageTokenizer(
            mode="DI",
            spatial_compression=16,
            max_resolution=4096,
        )

        assert tokenizer.modality == Modality.IMAGE
        assert tokenizer.mode == "DI"

    def test_metadata(self):
        """Test metadata retrieval."""
        tokenizer = EnterpriseImageTokenizer(
            mode="DI",
            spatial_compression=16,
        )

        metadata = tokenizer.get_metadata()

        assert metadata["modality"] == "image"
        assert metadata["mode"] == "DI"
        assert metadata["spatial_compression"] == 16

    def test_token_count(self):
        """Test token count estimation."""
        tokenizer = EnterpriseImageTokenizer(
            mode="DI",
            spatial_compression=16,
        )

        image = torch.randn(1, 3, 256, 256)
        token_count = tokenizer.get_token_count(image)

        # (256/16)^2 = 256 tokens
        assert token_count == 256

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        tokenizer = EnterpriseImageTokenizer(
            mode="DI",
            spatial_compression=16,
        )

        image = torch.randn(1, 3, 256, 256)
        ratio = tokenizer.get_compression_ratio(image)

        # Original: 1*3*256*256 = 196,608 elements
        # Tokenized: 1*16*16 = 256 elements
        # Ratio: 196,608 / 256 = 768
        assert ratio == pytest.approx(768.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

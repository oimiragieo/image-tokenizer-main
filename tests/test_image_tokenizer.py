"""Tests for image tokenizer API."""

import pytest
import torch
from cosmos_tokenizer import ImageTokenizer


class TestImageTokenizer:
    @pytest.mark.parametrize("mode", ["CI", "DI"])
    @pytest.mark.parametrize("spatial_compression", [8, 16])
    def test_initialization(self, mode, spatial_compression):
        tokenizer = ImageTokenizer(
            mode=mode,
            spatial_compression=spatial_compression,
            device="cpu",
        )

        assert tokenizer.config.mode == mode
        assert tokenizer.config.spatial_compression == spatial_compression

    def test_continuous_encode_decode(self):
        tokenizer = ImageTokenizer(mode="CI", spatial_compression=8, device="cpu")
        image = torch.randn(1, 3, 64, 64)

        # Encode
        latent = tokenizer.encode(image, deterministic=True)
        assert latent.shape[0] == 1
        assert latent.shape[1] == 16  # z_channels

        # Decode
        reconstructed = tokenizer.decode(latent)
        assert reconstructed.shape == image.shape

    def test_discrete_encode_decode(self):
        tokenizer = ImageTokenizer(mode="DI", spatial_compression=8, device="cpu")
        image = torch.randn(1, 3, 64, 64)

        # Encode
        indices = tokenizer.encode(image)
        assert indices.dtype in [torch.long, torch.int32, torch.int64]

        # Decode
        reconstructed = tokenizer.decode(indices)
        assert reconstructed.shape == image.shape

    def test_forward(self):
        tokenizer = ImageTokenizer(mode="CI", spatial_compression=8, device="cpu")
        image = torch.randn(2, 3, 64, 64)

        reconstructed = tokenizer(image)

        assert reconstructed.shape == image.shape

    def test_compression_ratio(self):
        tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cpu")
        ratio = tokenizer.get_compression_ratio()

        assert ratio == 256  # 16^2

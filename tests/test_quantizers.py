"""Tests for quantization modules."""

import pytest
import torch
from cosmos_tokenizer.modules.quantizers import (
    VectorQuantizer,
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
)


class TestVectorQuantizer:
    def test_forward(self):
        quantizer = VectorQuantizer(num_embeddings=128, embedding_dim=16)
        z = torch.randn(2, 16, 8, 8)

        z_q, indices, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert indices.shape == (2, 8, 8)
        assert isinstance(loss, torch.Tensor)
        assert "perplexity" in info

    def test_embed_code(self):
        quantizer = VectorQuantizer(num_embeddings=128, embedding_dim=16)
        indices = torch.randint(0, 128, (2, 8, 8))

        embeddings = quantizer.embed_code(indices)

        assert embeddings.shape == (2, 8, 8, 16)


class TestFSQuantizer:
    def test_forward(self):
        quantizer = FSQuantizer(levels=[8, 8, 8, 5, 5, 5])
        z = torch.randn(2, 6, 8, 8)

        z_q, indices, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert indices.shape == (2, 8, 8)
        assert quantizer.codebook_size == 64000

    def test_codebook_size(self):
        quantizer = FSQuantizer(levels=[8, 8])
        assert quantizer.codebook_size == 64


class TestLFQuantizer:
    def test_forward(self):
        quantizer = LFQuantizer(embedding_dim=16)
        z = torch.randn(2, 16, 8, 8)

        z_q, indices, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        # Quantized values should be -1 or 1
        assert torch.all((z_q == 1.0) | (z_q == -1.0))


class TestResidualFSQuantizer:
    def test_forward(self):
        quantizer = ResidualFSQuantizer(num_stages=2, levels=[8, 8])
        z = torch.randn(2, 2, 8, 8)

        z_q, indices, loss, info = quantizer(z)

        assert z_q.shape == z.shape
        assert indices.shape == (2, 8, 8, 2)  # 2 stages

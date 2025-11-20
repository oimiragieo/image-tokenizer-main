"""
Cosmos Image Tokenizer
======================

A world-class image and video tokenizer integrating state-of-the-art compression techniques.

Key Features:
- Dual-codebook architecture (semantic + visual)
- Object-aware compression with SAM integration
- Continuous and discrete modes
- Causal video processing
- Multi-stage quantization (VQ, FSQ, LFQ, Residual FSQ)
- Up to 2048× compression ratio

Usage:
    >>> from cosmos_tokenizer import ImageTokenizer
    >>> tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
    >>> latent = tokenizer.encode(image)
    >>> reconstructed = tokenizer.decode(latent)
"""

__version__ = "1.0.0"
__author__ = "Cosmos Tokenizer Team"
__license__ = "Apache-2.0"

from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.video_lib import VideoTokenizer, CausalVideoTokenizer
from cosmos_tokenizer.networks.configs import TokenizerConfig
from cosmos_tokenizer.modules.quantizers import (
    VectorQuantizer,
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
)

__all__ = [
    "ImageTokenizer",
    "VideoTokenizer",
    "CausalVideoTokenizer",
    "TokenizerConfig",
    "VectorQuantizer",
    "FSQuantizer",
    "LFQuantizer",
    "ResidualFSQuantizer",
]

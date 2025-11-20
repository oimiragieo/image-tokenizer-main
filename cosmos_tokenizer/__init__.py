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
- Enterprise-grade features (billing, safety, monitoring)
- Modality-aware architecture

Usage:
    >>> from cosmos_tokenizer import ImageTokenizer
    >>> tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
    >>> latent = tokenizer.encode(image)
    >>> reconstructed = tokenizer.decode(latent)

    # Enterprise features
    >>> from cosmos_tokenizer.enterprise import EnterpriseImageTokenizer
    >>> enterprise_tokenizer = EnterpriseImageTokenizer(
    ...     mode="DI", spatial_compression=16, max_resolution=4096
    ... )
    >>> result = enterprise_tokenizer.tokenize(image)
"""

__version__ = "1.0.0"
__author__ = "Cosmos Tokenizer Team"
__license__ = "Apache-2.0"

from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.modules.quantizers import (
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
    VectorQuantizer,
)
from cosmos_tokenizer.networks.configs import TokenizerConfig
from cosmos_tokenizer.video_lib import CausalVideoTokenizer, VideoTokenizer

# Enterprise features (optional imports)
try:
    from cosmos_tokenizer import benchmarks, core, enterprise

    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False

__all__ = [
    "ImageTokenizer",
    "VideoTokenizer",
    "CausalVideoTokenizer",
    "TokenizerConfig",
    "VectorQuantizer",
    "FSQuantizer",
    "LFQuantizer",
    "ResidualFSQuantizer",
    "ENTERPRISE_AVAILABLE",
]

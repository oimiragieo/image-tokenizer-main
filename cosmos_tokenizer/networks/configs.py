"""Configuration classes for tokenizer models."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TokenizerConfig:
    """
    Configuration for Cosmos Tokenizer models.

    Args:
        mode: Tokenizer mode (CI, DI, CV, DV)
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        z_channels: Number of latent channels
        ch: Base channel count
        ch_mult: Channel multipliers for each resolution
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
        spatial_compression: Spatial compression factor (8 or 16)
        temporal_compression: Temporal compression factor (4 or 8, for video)
        patch_size: Patch size (2 or 4)
        patch_method: Patching method ('haar' or 'rearrange')
        quantizer_type: Type of quantizer for discrete modes
        num_embeddings: Codebook size for VQ
        fsq_levels: FSQ levels per dimension
        use_dual_codebook: Whether to use dual codebook (semantic + visual)
    """

    # Model mode
    mode: Literal["CI", "DI", "CV", "DV"] = "DI"

    # Input/Output
    in_channels: int = 3
    out_channels: int = 3

    # Latent space
    z_channels: int = 16
    double_z: bool = True  # For continuous mode (mean + logvar)

    # Architecture
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_resolutions: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.0

    # Compression
    spatial_compression: int = 16
    temporal_compression: int = 8  # For video modes

    # Patching
    patch_size: int = 4
    patch_method: Literal["haar", "rearrange"] = "haar"

    # Quantization (for discrete modes)
    quantizer_type: Literal["vq", "fsq", "lfq", "residual_fsq"] = "fsq"
    num_embeddings: int = 64000  # For VQ
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    lfq_entropy_weight: float = 0.1
    residual_fsq_stages: int = 4

    # Dual codebook (GloTok-inspired)
    use_dual_codebook: bool = False
    semantic_channels: int = 6  # High-frequency components
    visual_channels: int = 6  # Low-frequency components

    # Training
    kl_weight: float = 1e-6  # For continuous mode
    commitment_cost: float = 0.25  # For VQ

    # Video-specific
    temporal_downsample_factors: List[bool] = field(default_factory=lambda: [False, True, True, False])
    temporal_upsample_factors: List[bool] = field(default_factory=lambda: [False, True, True, False])
    causal: bool = True  # For video modes

    def __post_init__(self):
        """Validate configuration."""
        assert self.mode in ["CI", "DI", "CV", "DV"], f"Invalid mode: {self.mode}"
        assert self.spatial_compression in [8, 16], f"Invalid spatial compression: {self.spatial_compression}"

        if self.mode in ["CV", "DV"]:
            assert self.temporal_compression in [4, 8], f"Invalid temporal compression: {self.temporal_compression}"

        # Adjust z_channels for discrete mode with quantizer
        if self.mode in ["DI", "DV"]:
            if self.quantizer_type == "fsq":
                self.z_channels = len(self.fsq_levels)
            elif self.quantizer_type == "lfq":
                self.z_channels = 16  # Default for binary quantization
            elif self.quantizer_type == "vq":
                self.z_channels = 16  # Embedding dimension

        # For continuous mode, double z_channels for mean + logvar
        if self.mode in ["CI", "CV"] and self.double_z:
            self.quant_channels = 2 * self.z_channels
        else:
            self.quant_channels = self.z_channels


# Pre-configured model variants
COSMOS_CI_8x8 = TokenizerConfig(
    mode="CI",
    spatial_compression=8,
    z_channels=16,
    ch=128,
    patch_size=4,
)

COSMOS_CI_16x16 = TokenizerConfig(
    mode="CI",
    spatial_compression=16,
    z_channels=16,
    ch=128,
    patch_size=4,
)

COSMOS_DI_8x8 = TokenizerConfig(
    mode="DI",
    spatial_compression=8,
    z_channels=6,
    ch=128,
    patch_size=4,
    quantizer_type="fsq",
    fsq_levels=[8, 8, 8, 5, 5, 5],
)

COSMOS_DI_16x16 = TokenizerConfig(
    mode="DI",
    spatial_compression=16,
    z_channels=6,
    ch=128,
    patch_size=4,
    quantizer_type="fsq",
    fsq_levels=[8, 8, 8, 5, 5, 5],
)

COSMOS_CV_8x8x8 = TokenizerConfig(
    mode="CV",
    spatial_compression=8,
    temporal_compression=8,
    z_channels=16,
    ch=128,
    patch_size=4,
    causal=True,
)

COSMOS_CV_4x8x8 = TokenizerConfig(
    mode="CV",
    spatial_compression=8,
    temporal_compression=4,
    z_channels=16,
    ch=128,
    patch_size=4,
    causal=True,
)

COSMOS_DV_8x16x16 = TokenizerConfig(
    mode="DV",
    spatial_compression=16,
    temporal_compression=8,
    z_channels=6,
    ch=128,
    patch_size=4,
    quantizer_type="residual_fsq",
    residual_fsq_stages=4,
    causal=True,
)


# Configuration registry
MODEL_CONFIGS = {
    "ci_8x8": COSMOS_CI_8x8,
    "ci_16x16": COSMOS_CI_16x16,
    "di_8x8": COSMOS_DI_8x8,
    "di_16x16": COSMOS_DI_16x16,
    "cv_8x8x8": COSMOS_CV_8x8x8,
    "cv_4x8x8": COSMOS_CV_4x8x8,
    "dv_8x16x16": COSMOS_DV_8x16x16,
}


def get_config(name: str) -> TokenizerConfig:
    """Get pre-configured model config by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown config: {name}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[name]

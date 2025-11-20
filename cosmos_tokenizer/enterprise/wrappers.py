"""
Enterprise wrappers for existing tokenizers to support modality-aware interface.

These wrappers integrate the original ImageTokenizer and VideoTokenizer with:
- ModalityAwareTokenizer base interface
- Safety validation
- Token counting and billing hooks
- Performance metrics tracking
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch

from cosmos_tokenizer.core.modality import (
    DetokenizationResult,
    Modality,
    ModalityAwareTokenizer,
    TokenizationResult,
)
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.video_lib import VideoTokenizer


class EnterpriseImageTokenizer(ModalityAwareTokenizer):
    """
    Enterprise-grade wrapper for ImageTokenizer with modality-aware interface.

    Features:
    - Safety validation (file size, resolution limits)
    - Token counting for billing
    - Performance metrics tracking
    - Standardized API (tokenize/detokenize)
    """

    def __init__(
        self,
        mode: str = "DI",
        spatial_compression: int = 16,
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "float32",
        version: str = "1.0.0",
        max_resolution: int = 4096,
        max_file_size_mb: float = 100.0,
        **kwargs,
    ):
        """
        Args:
            mode: Tokenizer mode (CI=continuous, DI=discrete)
            spatial_compression: Spatial compression factor (8 or 16)
            checkpoint: Path to checkpoint file
            device: Device to load model on
            dtype: Data type (float32, float16, bfloat16)
            version: Tokenizer version
            max_resolution: Maximum image resolution (width or height)
            max_file_size_mb: Maximum file size in MB
            **kwargs: Additional configuration
        """
        super().__init__(
            modality=Modality.IMAGE,
            version=version,
            device=device,
        )

        # Initialize underlying tokenizer
        self._tokenizer = ImageTokenizer(
            mode=mode,
            spatial_compression=spatial_compression,
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.spatial_compression = spatial_compression
        self.max_resolution = max_resolution
        self.max_file_size_mb = max_file_size_mb

    def tokenize(
        self,
        input_data: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> TokenizationResult:
        """
        Tokenize image into discrete or continuous tokens.

        Args:
            input_data: Input image (B, 3, H, W), range [-1, 1]
            deterministic: For CI mode, use mean without sampling
            **kwargs: Additional options

        Returns:
            TokenizationResult with tokens and metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")

        # Encode
        tokens = self._tokenizer.encode(input_data, deterministic=deterministic)

        # Calculate token count
        token_count = tokens.numel()

        # Calculate compression ratio
        compression_ratio = self.get_compression_ratio(input_data)

        # Build metadata
        metadata = {
            "mode": self.mode,
            "spatial_compression": self.spatial_compression,
            "compression_ratio": compression_ratio,
            "input_shape": tuple(input_data.shape),
            "output_shape": tuple(tokens.shape),
            "deterministic": deterministic,
        }

        # Update stats
        self._update_stats("tokenize", token_count)

        return TokenizationResult(
            tokens=tokens,
            modality=self.modality,
            token_count=token_count,
            metadata=metadata,
            version=self.version,
        )

    def detokenize(
        self,
        tokens: torch.Tensor,
        **kwargs,
    ) -> DetokenizationResult:
        """
        Reconstruct image from tokens.

        Args:
            tokens: Tokens to decode
            **kwargs: Additional options

        Returns:
            DetokenizationResult with reconstructed image
        """
        # Decode
        output = self._tokenizer.decode(tokens)

        # Build metadata
        metadata = {
            "mode": self.mode,
            "input_shape": tuple(tokens.shape),
            "output_shape": tuple(output.shape),
        }

        # Update stats
        self._update_stats("detokenize")

        return DetokenizationResult(
            output=output,
            modality=self.modality,
            metadata=metadata,
            version=self.version,
        )

    def validate_input(
        self,
        input_data: torch.Tensor,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate image input.

        Checks:
        - Tensor shape (B, 3, H, W)
        - Resolution limits
        - Value range [-1, 1]
        - Divisibility by compression factor

        Args:
            input_data: Input to validate

        Returns:
            (is_valid, error_message) tuple
        """
        # Check tensor
        if not isinstance(input_data, torch.Tensor):
            return False, "Input must be a torch.Tensor"

        # Check shape
        if input_data.ndim != 4:
            return False, f"Expected 4D tensor (B,C,H,W), got {input_data.ndim}D"

        batch, channels, height, width = input_data.shape

        if channels != 3:
            return False, f"Expected 3 channels, got {channels}"

        # Check resolution
        if max(height, width) > self.max_resolution:
            return (
                False,
                f"Resolution {height}x{width} exceeds max {self.max_resolution}",
            )

        # Check value range (with some tolerance)
        min_val, max_val = input_data.min().item(), input_data.max().item()
        if min_val < -1.1 or max_val > 1.1:
            return (
                False,
                f"Values must be in [-1, 1], got [{min_val:.2f}, {max_val:.2f}]",
            )

        # Check divisibility
        patch_size = getattr(self._tokenizer.config, "patch_size", 1)
        total_compression = self.spatial_compression * patch_size

        if height % total_compression != 0 or width % total_compression != 0:
            return (
                False,
                f"Dimensions must be divisible by {total_compression}, "
                f"got {height}x{width}",
            )

        return True, None

    def get_token_count(self, input_data: torch.Tensor) -> int:
        """
        Calculate token count without full tokenization.

        Args:
            input_data: Input image (B, 3, H, W)

        Returns:
            Expected token count
        """
        batch, _, height, width = input_data.shape
        h = height // self.spatial_compression
        w = width // self.spatial_compression

        if self.mode == "DI":
            # Discrete: (B, h, w)
            return batch * h * w
        else:
            # Continuous: (B, z_channels, h, w)
            z_channels = getattr(self._tokenizer.config, "z_channels", 16)
            return batch * z_channels * h * w

    def get_compression_ratio(self, input_data: torch.Tensor) -> float:
        """
        Calculate compression ratio.

        Args:
            input_data: Input image

        Returns:
            Compression ratio (original_size / tokenized_size)
        """
        original_size = input_data.numel()
        token_count = self.get_token_count(input_data)
        return original_size / token_count

    def get_metadata(self) -> Dict[str, Any]:
        """Get tokenizer metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "mode": self.mode,
                "spatial_compression": self.spatial_compression,
                "max_resolution": self.max_resolution,
                "max_file_size_mb": self.max_file_size_mb,
                "codebook_size": self._tokenizer.get_codebook_size(),
            }
        )
        return metadata


class EnterpriseVideoTokenizer(ModalityAwareTokenizer):
    """
    Enterprise-grade wrapper for VideoTokenizer with modality-aware interface.

    Features:
    - Safety validation (file size, resolution, frame count limits)
    - Token counting for billing
    - Performance metrics tracking
    - Standardized API (tokenize/detokenize)
    """

    def __init__(
        self,
        mode: str = "DV",
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "float32",
        version: str = "1.0.0",
        max_resolution: int = 2048,
        max_frames: int = 1000,
        max_file_size_mb: float = 500.0,
        **kwargs,
    ):
        """
        Args:
            mode: Tokenizer mode (CV=continuous, DV=discrete)
            spatial_compression: Spatial compression factor (8 or 16)
            temporal_compression: Temporal compression factor (4 or 8)
            checkpoint: Path to checkpoint file
            device: Device to load model on
            dtype: Data type (float32, float16, bfloat16)
            version: Tokenizer version
            max_resolution: Maximum video resolution (width or height)
            max_frames: Maximum number of frames
            max_file_size_mb: Maximum file size in MB
            **kwargs: Additional configuration
        """
        super().__init__(
            modality=Modality.VIDEO,
            version=version,
            device=device,
        )

        # Initialize underlying tokenizer
        self._tokenizer = VideoTokenizer(
            mode=mode,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression
        self.max_resolution = max_resolution
        self.max_frames = max_frames
        self.max_file_size_mb = max_file_size_mb

    def tokenize(
        self,
        input_data: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> TokenizationResult:
        """
        Tokenize video into discrete or continuous tokens.

        Args:
            input_data: Input video (B, 3, T, H, W), range [-1, 1]
            deterministic: For CV mode, use mean without sampling
            **kwargs: Additional options

        Returns:
            TokenizationResult with tokens and metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {error_msg}")

        # Encode
        tokens = self._tokenizer.encode(input_data, deterministic=deterministic)

        # Calculate token count
        token_count = tokens.numel()

        # Calculate compression ratio
        compression_ratio = self.get_compression_ratio(input_data)

        # Build metadata
        metadata = {
            "mode": self.mode,
            "spatial_compression": self.spatial_compression,
            "temporal_compression": self.temporal_compression,
            "compression_ratio": compression_ratio,
            "input_shape": tuple(input_data.shape),
            "output_shape": tuple(tokens.shape),
            "deterministic": deterministic,
        }

        # Update stats
        self._update_stats("tokenize", token_count)

        return TokenizationResult(
            tokens=tokens,
            modality=self.modality,
            token_count=token_count,
            metadata=metadata,
            version=self.version,
        )

    def detokenize(
        self,
        tokens: torch.Tensor,
        **kwargs,
    ) -> DetokenizationResult:
        """
        Reconstruct video from tokens.

        Args:
            tokens: Tokens to decode
            **kwargs: Additional options

        Returns:
            DetokenizationResult with reconstructed video
        """
        # Decode
        output = self._tokenizer.decode(tokens)

        # Build metadata
        metadata = {
            "mode": self.mode,
            "input_shape": tuple(tokens.shape),
            "output_shape": tuple(output.shape),
        }

        # Update stats
        self._update_stats("detokenize")

        return DetokenizationResult(
            output=output,
            modality=self.modality,
            metadata=metadata,
            version=self.version,
        )

    def validate_input(
        self,
        input_data: torch.Tensor,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate video input.

        Checks:
        - Tensor shape (B, 3, T, H, W)
        - Resolution limits
        - Frame count limits
        - Value range [-1, 1]
        - Divisibility by compression factors

        Args:
            input_data: Input to validate

        Returns:
            (is_valid, error_message) tuple
        """
        # Check tensor
        if not isinstance(input_data, torch.Tensor):
            return False, "Input must be a torch.Tensor"

        # Check shape
        if input_data.ndim != 5:
            return False, f"Expected 5D tensor (B,C,T,H,W), got {input_data.ndim}D"

        batch, channels, frames, height, width = input_data.shape

        if channels != 3:
            return False, f"Expected 3 channels, got {channels}"

        # Check resolution
        if max(height, width) > self.max_resolution:
            return (
                False,
                f"Resolution {height}x{width} exceeds max {self.max_resolution}",
            )

        # Check frame count
        if frames > self.max_frames:
            return False, f"Frame count {frames} exceeds max {self.max_frames}"

        # Check value range (with some tolerance)
        min_val, max_val = input_data.min().item(), input_data.max().item()
        if min_val < -1.1 or max_val > 1.1:
            return (
                False,
                f"Values must be in [-1, 1], got [{min_val:.2f}, {max_val:.2f}]",
            )

        return True, None

    def get_token_count(self, input_data: torch.Tensor) -> int:
        """
        Calculate token count without full tokenization.

        Args:
            input_data: Input video (B, 3, T, H, W)

        Returns:
            Expected token count
        """
        batch, _, frames, height, width = input_data.shape
        t = frames // self.temporal_compression
        h = height // self.spatial_compression
        w = width // self.spatial_compression

        if self.mode == "DV":
            # Discrete: (B, t, h, w)
            return batch * t * h * w
        else:
            # Continuous: (B, z_channels, t, h, w)
            z_channels = getattr(self._tokenizer.config, "z_channels", 16)
            return batch * z_channels * t * h * w

    def get_compression_ratio(self, input_data: torch.Tensor) -> float:
        """
        Calculate compression ratio.

        Args:
            input_data: Input video

        Returns:
            Compression ratio (original_size / tokenized_size)
        """
        original_size = input_data.numel()
        token_count = self.get_token_count(input_data)
        return original_size / token_count

    def get_metadata(self) -> Dict[str, Any]:
        """Get tokenizer metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "mode": self.mode,
                "spatial_compression": self.spatial_compression,
                "temporal_compression": self.temporal_compression,
                "max_resolution": self.max_resolution,
                "max_frames": self.max_frames,
                "max_file_size_mb": self.max_file_size_mb,
                "codebook_size": self._tokenizer.get_codebook_size(),
            }
        )
        return metadata

"""High-level API for image tokenization."""

import torch
import torch.nn as nn
from typing import Union, Optional, Literal
from pathlib import Path

from cosmos_tokenizer.networks.configs import TokenizerConfig, get_config
from cosmos_tokenizer.networks.autoencoder import (
    ContinuousImageTokenizer,
    DiscreteImageTokenizer,
)


class ImageTokenizer(nn.Module):
    """
    Unified high-level API for image tokenization.
    
    Supports both continuous (CI) and discrete (DI) modes.
    
    Example:
        >>> tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
        >>> image = torch.randn(1, 3, 256, 256)  # Range [-1, 1]
        >>> latent = tokenizer.encode(image)
        >>> reconstructed = tokenizer.decode(latent)
    """

    def __init__(
        self,
        mode: Literal["CI", "DI"] = "DI",
        spatial_compression: int = 16,
        config: Optional[TokenizerConfig] = None,
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Union[str, torch.dtype] = "float32",
    ):
        """
        Args:
            mode: Tokenizer mode (CI=continuous, DI=discrete)
            spatial_compression: Spatial compression factor (8 or 16)
            config: Custom configuration (overrides mode/spatial_compression)
            checkpoint: Path to checkpoint file
            device: Device to load model on
            dtype: Data type (float32, float16, bfloat16)
        """
        super().__init__()
        
        # Get configuration
        if config is None:
            config_name = f"{mode.lower()}_{spatial_compression}x{spatial_compression}"
            config = get_config(config_name)
        
        self.config = config
        self.device = torch.device(device)
        
        # Parse dtype
        if isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(dtype, torch.float32)
        else:
            self.dtype = dtype
        
        # Create model
        if config.mode == "CI":
            self.model = ContinuousImageTokenizer(config)
        elif config.mode == "DI":
            self.model = DiscreteImageTokenizer(config)
        else:
            raise ValueError(f"Invalid mode for ImageTokenizer: {config.mode}")
        
        # Load checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        
        # Move to device and set dtype
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    def encode(
        self,
        image: torch.Tensor,
        deterministic: bool = False,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Encode image to latent representation.
        
        Args:
            image: Input image (B, 3, H, W), range [-1, 1]
            deterministic: For CI mode, use mean without sampling
            return_dict: If True, return full output dict
            
        Returns:
            For DI mode: Discrete indices (B, h, w)
            For CI mode: Continuous latent (B, z_channels, h, w)
            If return_dict=True: Full EncoderOutput
        """
        image = image.to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            if self.config.mode == "CI":
                output = self.model.encode(image, deterministic=deterministic)
            else:
                output = self.model.encode(image)
        
        if return_dict:
            return output
        else:
            return output.indices if output.indices is not None else output.latent

    def decode(
        self,
        latent: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Decode latent to image.
        
        Args:
            latent: Latent representation or indices
            return_dict: If True, return full output dict
            
        Returns:
            Reconstructed image (B, 3, H, W), range [-1, 1]
            If return_dict=True: Full DecoderOutput
        """
        latent = latent.to(self.device)
        
        with torch.no_grad():
            # For discrete mode with indices, use decode_indices
            if self.config.mode == "DI" and latent.dtype in [torch.long, torch.int32, torch.int64]:
                output = self.model.decode_indices(latent)
                if return_dict:
                    from cosmos_tokenizer.networks.autoencoder import DecoderOutput
                    output = DecoderOutput(reconstruction=output)
            else:
                latent = latent.to(dtype=self.dtype)
                output = self.model.decode(latent)
        
        if return_dict:
            return output
        else:
            return output.reconstruction if hasattr(output, 'reconstruction') else output

    def forward(
        self,
        image: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Full round-trip: encode and decode.
        
        Args:
            image: Input image (B, 3, H, W), range [-1, 1]
            deterministic: For CI mode, use mean without sampling
            
        Returns:
            Reconstructed image
        """
        image = image.to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            reconstruction, info = self.model(image, deterministic=deterministic)
        
        return reconstruction

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.suffix == ".jit":
            # Load TorchScript model
            self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        else:
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            
            self.model.load_state_dict(state_dict, strict=True)

    def save_checkpoint(self, checkpoint_path: str):
        """Save model weights to checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": self.config,
        }, checkpoint_path)

    def to_jit(self) -> torch.jit.ScriptModule:
        """Convert model to TorchScript for optimized inference."""
        example_input = torch.randn(1, 3, 256, 256, device=self.device, dtype=self.dtype)
        return torch.jit.trace(self.model, example_input)

    @torch.no_grad()
    def autoencode(self, image: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(image, deterministic=deterministic)

    def get_codebook_size(self) -> Optional[int]:
        """Get codebook size for discrete tokenizers."""
        if self.config.mode == "DI":
            return self.model.quantizer.codebook_size
        return None

    def get_compression_ratio(self) -> float:
        """Get total compression ratio."""
        spatial = self.config.spatial_compression ** 2
        if self.config.mode == "DI":
            # Discrete: compression in tokens
            return spatial
        else:
            # Continuous: compression in spatial dimensions
            return spatial

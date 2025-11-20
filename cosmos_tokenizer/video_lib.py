"""High-level API for video tokenization."""

import torch
import torch.nn as nn
from typing import Union, Optional, Literal
from pathlib import Path

from cosmos_tokenizer.networks.configs import TokenizerConfig, get_config
from cosmos_tokenizer.networks.autoencoder import (
    ContinuousVideoTokenizer,
    DiscreteVideoTokenizer,
)


class VideoTokenizer(nn.Module):
    """
    Unified high-level API for video tokenization.
    
    Supports both continuous (CV) and discrete (DV) modes with causal processing.
    
    Example:
        >>> tokenizer = VideoTokenizer(mode="CV", spatial_compression=8, temporal_compression=8)
        >>> video = torch.randn(1, 3, 32, 256, 256)  # Range [-1, 1]
        >>> latent = tokenizer.encode(video)
        >>> reconstructed = tokenizer.decode(latent)
    """

    def __init__(
        self,
        mode: Literal["CV", "DV"] = "CV",
        spatial_compression: int = 8,
        temporal_compression: int = 8,
        config: Optional[TokenizerConfig] = None,
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: Union[str, torch.dtype] = "float32",
    ):
        """
        Args:
            mode: Tokenizer mode (CV=continuous, DV=discrete)
            spatial_compression: Spatial compression factor (8 or 16)
            temporal_compression: Temporal compression factor (4 or 8)
            config: Custom configuration (overrides other params)
            checkpoint: Path to checkpoint file
            device: Device to load model on
            dtype: Data type (float32, float16, bfloat16)
        """
        super().__init__()
        
        # Get configuration
        if config is None:
            config_name = f"{mode.lower()}_{temporal_compression}x{spatial_compression}x{spatial_compression}"
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
        if config.mode == "CV":
            self.model = ContinuousVideoTokenizer(config)
        elif config.mode == "DV":
            self.model = DiscreteVideoTokenizer(config)
        else:
            raise ValueError(f"Invalid mode for VideoTokenizer: {config.mode}")
        
        # Load checkpoint if provided
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        
        # Move to device and set dtype
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    def encode(
        self,
        video: torch.Tensor,
        deterministic: bool = False,
        temporal_window: Optional[int] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Encode video to latent representation.
        
        Args:
            video: Input video (B, 3, T, H, W), range [-1, 1]
            deterministic: For CV mode, use mean without sampling
            temporal_window: Process video in windows (for long videos)
            return_dict: If True, return full output dict
            
        Returns:
            For DV mode: Discrete indices (B, t, h, w)
            For CV mode: Continuous latent (B, z_channels, t, h, w)
        """
        video = video.to(self.device, dtype=self.dtype)
        
        # Handle long videos with sliding window
        if temporal_window is not None and video.shape[2] > temporal_window:
            return self._encode_sliding_window(
                video, deterministic, temporal_window, return_dict
            )
        
        with torch.no_grad():
            if self.config.mode == "CV":
                output = self.model.encode(video, deterministic=deterministic)
            else:
                output = self.model.encode(video)
        
        if return_dict:
            return output
        else:
            return output.indices if output.indices is not None else output.latent

    def decode(
        self,
        latent: torch.Tensor,
        temporal_window: Optional[int] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Decode latent to video.
        
        Args:
            latent: Latent representation or indices
            temporal_window: Process in windows (for long videos)
            return_dict: If True, return full output dict
            
        Returns:
            Reconstructed video (B, 3, T, H, W), range [-1, 1]
        """
        latent = latent.to(self.device)
        
        # Handle long videos with sliding window
        if temporal_window is not None and latent.shape[2] > temporal_window:
            return self._decode_sliding_window(latent, temporal_window, return_dict)
        
        with torch.no_grad():
            latent = latent.to(dtype=self.dtype)
            output = self.model.decode(latent)
        
        if return_dict:
            return output
        else:
            return output.reconstruction if hasattr(output, 'reconstruction') else output

    def forward(
        self,
        video: torch.Tensor,
        deterministic: bool = False,
        temporal_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Full round-trip: encode and decode.
        
        Args:
            video: Input video (B, 3, T, H, W), range [-1, 1]
            deterministic: For CV mode, use mean without sampling
            temporal_window: Process in windows (recommended: 17)
            
        Returns:
            Reconstructed video
        """
        video = video.to(self.device, dtype=self.dtype)
        
        # Handle long videos with sliding window
        if temporal_window is not None and video.shape[2] > temporal_window:
            latent = self.encode(video, deterministic, temporal_window)
            return self.decode(latent, temporal_window)
        
        with torch.no_grad():
            reconstruction, info = self.model(video, deterministic=deterministic)
        
        return reconstruction

    def _encode_sliding_window(
        self,
        video: torch.Tensor,
        deterministic: bool,
        window_size: int,
        return_dict: bool,
    ):
        """Encode long video using sliding window."""
        B, C, T, H, W = video.shape
        
        # Calculate overlap (50% for smooth transitions)
        stride = window_size // 2
        
        latents = []
        
        for start_idx in range(0, T, stride):
            end_idx = min(start_idx + window_size, T)
            
            # Extract window
            window = video[:, :, start_idx:end_idx, :, :]
            
            # Pad if necessary
            if window.shape[2] < window_size:
                pad_size = window_size - window.shape[2]
                window = torch.cat([
                    window,
                    window[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
                ], dim=2)
            
            # Encode window
            with torch.no_grad():
                if self.config.mode == "CV":
                    output = self.model.encode(window, deterministic=deterministic)
                else:
                    output = self.model.encode(window)
            
            latent_window = output.indices if output.indices is not None else output.latent
            
            # Remove padding
            if end_idx - start_idx < window_size:
                actual_frames = end_idx - start_idx
                latent_frames = actual_frames // self.config.temporal_compression
                latent_window = latent_window[:, :, :latent_frames, :, :]
            
            latents.append(latent_window)
        
        # Concatenate (handle overlap by averaging)
        result = torch.cat(latents, dim=2)
        
        if return_dict:
            from cosmos_tokenizer.networks.autoencoder import EncoderOutput
            return EncoderOutput(latent=result)
        return result

    def _decode_sliding_window(
        self,
        latent: torch.Tensor,
        window_size: int,
        return_dict: bool,
    ):
        """Decode long latent using sliding window."""
        B, C, T, H, W = latent.shape
        
        stride = window_size // 2
        reconstructions = []
        
        for start_idx in range(0, T, stride):
            end_idx = min(start_idx + window_size, T)
            
            # Extract window
            window = latent[:, :, start_idx:end_idx, :, :]
            
            # Pad if necessary
            if window.shape[2] < window_size:
                pad_size = window_size - window.shape[2]
                window = torch.cat([
                    window,
                    window[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
                ], dim=2)
            
            # Decode window
            with torch.no_grad():
                output = self.model.decode(window.to(dtype=self.dtype))
            
            recon_window = output.reconstruction if hasattr(output, 'reconstruction') else output
            
            # Remove padding
            if end_idx - start_idx < window_size:
                actual_frames = (end_idx - start_idx) * self.config.temporal_compression
                recon_window = recon_window[:, :, :actual_frames, :, :]
            
            reconstructions.append(recon_window)
        
        # Concatenate
        result = torch.cat(reconstructions, dim=2)
        
        if return_dict:
            from cosmos_tokenizer.networks.autoencoder import DecoderOutput
            return DecoderOutput(reconstruction=result)
        return result

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.suffix == ".jit":
            self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        else:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
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

    @torch.no_grad()
    def autoencode(
        self,
        video: torch.Tensor,
        deterministic: bool = False,
        temporal_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(video, deterministic, temporal_window)


class CausalVideoTokenizer(VideoTokenizer):
    """
    Causal Video Tokenizer with explicit temporal windowing.
    
    Optimized for autoregressive video generation with proper causal masking.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ensure causal mode is enabled
        if not self.config.causal:
            raise ValueError("CausalVideoTokenizer requires causal=True in config")

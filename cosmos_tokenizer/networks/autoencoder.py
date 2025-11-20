"""
Complete autoencoder implementations for all tokenizer modes.

Implements four variants:
- ContinuousImageTokenizer (CI): VAE-style for diffusion models
- DiscreteImageTokenizer (DI): VQ-style for autoregressive models
- ContinuousVideoTokenizer (CV): Causal VAE for video diffusion
- DiscreteVideoTokenizer (DV): Causal VQ for video autoregressive
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union, NamedTuple
import math

from cosmos_tokenizer.networks.configs import TokenizerConfig
from cosmos_tokenizer.modules.layers2d import Encoder2D, Decoder2D
from cosmos_tokenizer.modules.layers3d import Encoder3D, Decoder3D
from cosmos_tokenizer.modules.quantizers import (
    VectorQuantizer,
    FSQuantizer,
    LFQuantizer,
    ResidualFSQuantizer,
)
from cosmos_tokenizer.modules.distributions import DiagonalGaussianDistribution
from cosmos_tokenizer.modules.patching import (
    HaarPatch2D,
    HaarUnpatch2D,
    HaarPatch3D,
    HaarUnpatch3D,
    RearrangePatch,
    RearrangeUnpatch,
)


class EncoderOutput(NamedTuple):
    """Output from encoder."""
    latent: torch.Tensor
    indices: Optional[torch.Tensor] = None
    posterior: Optional[DiagonalGaussianDistribution] = None
    info: Dict = {}


class DecoderOutput(NamedTuple):
    """Output from decoder."""
    reconstruction: torch.Tensor
    info: Dict = {}


class ContinuousImageTokenizer(nn.Module):
    """
    Continuous Image Tokenizer (CI mode).
    
    Uses VAE-style encoding with Gaussian posterior for diffusion models.
    Achieves 8x8 or 16x16 spatial compression.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        assert config.mode == "CI", f"Expected CI mode, got {config.mode}"
        self.config = config

        # Patching
        if config.patch_method == "haar":
            self.patcher = HaarPatch2D(patch_size=2)
            self.unpatcher = HaarUnpatch2D(patch_size=2)
            patch_channels = config.in_channels * 4
        else:
            self.patcher = RearrangePatch(patch_size=config.patch_size)
            self.unpatcher = RearrangeUnpatch(patch_size=config.patch_size)
            patch_channels = config.in_channels * (config.patch_size ** 2)

        # Encoder
        self.encoder = Encoder2D(
            in_channels=patch_channels,
            z_channels=config.quant_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
        )

        # No quantization for continuous mode, just distribution
        self.kl_weight = config.kl_weight

        # Decoder
        self.decoder = Decoder2D(
            out_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
        )

    def encode(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> EncoderOutput:
        """
        Encode image to latent distribution.
        
        Args:
            x: Input image (B, 3, H, W), range [-1, 1]
            deterministic: If True, use mean without sampling
            
        Returns:
            EncoderOutput with latent and posterior
        """
        # Patch
        h = self.patcher(x)
        
        # Encode
        h = self.encoder(h)
        
        # Create distribution
        posterior = DiagonalGaussianDistribution(h, deterministic=deterministic)
        
        # Sample
        z = posterior.sample()
        
        # Compute KL loss
        kl_loss = posterior.kl() * self.kl_weight if not deterministic else 0.0
        
        info = {
            "kl_loss": kl_loss,
            "mean": posterior.mean,
            "logvar": posterior.logvar,
        }
        
        return EncoderOutput(latent=z, posterior=posterior, info=info)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """
        Decode latent to image.
        
        Args:
            z: Latent tensor (B, z_channels, h, w)
            
        Returns:
            DecoderOutput with reconstructed image
        """
        # Decode
        h = self.decoder(z)
        
        # Unpatch
        x_recon = self.unpatcher(h)
        
        return DecoderOutput(reconstruction=x_recon)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: encode and decode.
        
        Args:
            x: Input image
            deterministic: If True, use mean without sampling
            
        Returns:
            Reconstruction and info dictionary
        """
        enc_out = self.encode(x, deterministic=deterministic)
        dec_out = self.decode(enc_out.latent)
        
        info = {**enc_out.info, **dec_out.info}
        
        return dec_out.reconstruction, info


class DiscreteImageTokenizer(nn.Module):
    """
    Discrete Image Tokenizer (DI mode).
    
    Uses vector quantization for autoregressive models.
    Achieves 8x8 or 16x16 spatial compression with 64K vocabulary.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        assert config.mode == "DI", f"Expected DI mode, got {config.mode}"
        self.config = config

        # Patching
        if config.patch_method == "haar":
            self.patcher = HaarPatch2D(patch_size=2)
            self.unpatcher = HaarUnpatch2D(patch_size=2)
            patch_channels = config.in_channels * 4
        else:
            self.patcher = RearrangePatch(patch_size=config.patch_size)
            self.unpatcher = RearrangeUnpatch(patch_size=config.patch_size)
            patch_channels = config.in_channels * (config.patch_size ** 2)

        # Encoder
        self.encoder = Encoder2D(
            in_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
        )

        # Quantizer
        if config.quantizer_type == "vq":
            self.quantizer = VectorQuantizer(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.z_channels,
                commitment_cost=config.commitment_cost,
            )
        elif config.quantizer_type == "fsq":
            self.quantizer = FSQuantizer(
                levels=config.fsq_levels,
                embedding_dim=config.z_channels,
            )
        elif config.quantizer_type == "lfq":
            self.quantizer = LFQuantizer(
                embedding_dim=config.z_channels,
                entropy_loss_weight=config.lfq_entropy_weight,
            )
        elif config.quantizer_type == "residual_fsq":
            self.quantizer = ResidualFSQuantizer(
                num_stages=config.residual_fsq_stages,
                levels=config.fsq_levels,
                embedding_dim=config.z_channels,
            )
        else:
            raise ValueError(f"Unknown quantizer type: {config.quantizer_type}")

        # Decoder
        self.decoder = Decoder2D(
            out_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
        )

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """
        Encode image to discrete tokens.
        
        Args:
            x: Input image (B, 3, H, W), range [-1, 1]
            
        Returns:
            EncoderOutput with latent and indices
        """
        # Patch
        h = self.patcher(x)
        
        # Encode
        z = self.encoder(h)
        
        # Quantize
        z_q, indices, quant_loss, quant_info = self.quantizer(z)
        
        info = {
            "quantizer_loss": quant_loss,
            **quant_info,
        }
        
        return EncoderOutput(latent=z_q, indices=indices, info=info)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """
        Decode latent to image.
        
        Args:
            z: Latent tensor (B, z_channels, h, w)
            
        Returns:
            DecoderOutput with reconstructed image
        """
        # Decode
        h = self.decoder(z)
        
        # Unpatch
        x_recon = self.unpatcher(h)
        
        return DecoderOutput(reconstruction=x_recon)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete indices to image.
        
        Args:
            indices: Token indices (B, h, w)
            
        Returns:
            Reconstructed image
        """
        # Convert indices to embeddings
        z_q = self.quantizer.embed_code(indices)
        
        # If FSQ/LFQ, need to reshape properly
        if len(z_q.shape) == 3:  # (B, h*w, C)
            B, HW, C = z_q.shape
            h = w = int(math.sqrt(HW))
            z_q = z_q.reshape(B, h, w, C).permute(0, 3, 1, 2)
        
        # Decode
        dec_out = self.decode(z_q)
        
        return dec_out.reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: encode and decode.
        
        Args:
            x: Input image
            
        Returns:
            Reconstruction and info dictionary
        """
        enc_out = self.encode(x)
        dec_out = self.decode(enc_out.latent)
        
        info = {**enc_out.info, **dec_out.info}
        
        return dec_out.reconstruction, info


class ContinuousVideoTokenizer(nn.Module):
    """
    Continuous Video Tokenizer (CV mode).
    
    Uses causal VAE encoding for video diffusion models.
    Achieves 4x8x8 or 8x8x8 temporal-spatial compression.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        assert config.mode == "CV", f"Expected CV mode, got {config.mode}"
        self.config = config

        # Patching
        if config.patch_method == "haar":
            self.patcher = HaarPatch3D(patch_size=2, patch_temporal=True)
            self.unpatcher = HaarUnpatch3D(patch_size=2, patch_temporal=True)
            patch_channels = config.in_channels * 8  # 2^3 for 3D
        else:
            self.patcher = RearrangePatch(patch_size=config.patch_size)
            self.unpatcher = RearrangeUnpatch(patch_size=config.patch_size)
            patch_channels = config.in_channels * (config.patch_size ** 2)

        # Encoder (3D Causal)
        self.encoder = Encoder3D(
            in_channels=patch_channels,
            z_channels=config.quant_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            temporal_downsample_factors=config.temporal_downsample_factors,
        )

        self.kl_weight = config.kl_weight

        # Decoder (3D Causal)
        self.decoder = Decoder3D(
            out_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            temporal_upsample_factors=config.temporal_upsample_factors,
        )

    def encode(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> EncoderOutput:
        """
        Encode video to latent distribution.
        
        Args:
            x: Input video (B, 3, T, H, W), range [-1, 1]
            deterministic: If True, use mean without sampling
            
        Returns:
            EncoderOutput with latent and posterior
        """
        # Patch
        h = self.patcher(x)
        
        # Encode
        h = self.encoder(h)
        
        # Create distribution
        posterior = DiagonalGaussianDistribution(h, deterministic=deterministic)
        
        # Sample
        z = posterior.sample()
        
        # Compute KL loss
        kl_loss = posterior.kl() * self.kl_weight if not deterministic else 0.0
        
        info = {
            "kl_loss": kl_loss,
            "mean": posterior.mean,
            "logvar": posterior.logvar,
        }
        
        return EncoderOutput(latent=z, posterior=posterior, info=info)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """
        Decode latent to video.
        
        Args:
            z: Latent tensor (B, z_channels, t, h, w)
            
        Returns:
            DecoderOutput with reconstructed video
        """
        # Decode
        h = self.decoder(z)
        
        # Unpatch
        x_recon = self.unpatcher(h)
        
        return DecoderOutput(reconstruction=x_recon)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: encode and decode.
        
        Args:
            x: Input video
            deterministic: If True, use mean without sampling
            
        Returns:
            Reconstruction and info dictionary
        """
        enc_out = self.encode(x, deterministic=deterministic)
        dec_out = self.decode(enc_out.latent)
        
        info = {**enc_out.info, **dec_out.info}
        
        return dec_out.reconstruction, info


class DiscreteVideoTokenizer(nn.Module):
    """
    Discrete Video Tokenizer (DV mode).
    
    Uses causal vector quantization for autoregressive video models.
    Achieves 8x16x16 temporal-spatial compression with 64K vocabulary.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        assert config.mode == "DV", f"Expected DV mode, got {config.mode}"
        self.config = config

        # Patching
        if config.patch_method == "haar":
            self.patcher = HaarPatch3D(patch_size=2, patch_temporal=True)
            self.unpatcher = HaarUnpatch3D(patch_size=2, patch_temporal=True)
            patch_channels = config.in_channels * 8
        else:
            self.patcher = RearrangePatch(patch_size=config.patch_size)
            self.unpatcher = RearrangeUnpatch(patch_size=config.patch_size)
            patch_channels = config.in_channels * (config.patch_size ** 2)

        # Encoder (3D Causal)
        self.encoder = Encoder3D(
            in_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            temporal_downsample_factors=config.temporal_downsample_factors,
        )

        # Quantizer
        if config.quantizer_type == "vq":
            self.quantizer = VectorQuantizer(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.z_channels,
                commitment_cost=config.commitment_cost,
            )
        elif config.quantizer_type == "fsq":
            self.quantizer = FSQuantizer(
                levels=config.fsq_levels,
                embedding_dim=config.z_channels,
            )
        elif config.quantizer_type == "residual_fsq":
            self.quantizer = ResidualFSQuantizer(
                num_stages=config.residual_fsq_stages,
                levels=config.fsq_levels,
                embedding_dim=config.z_channels,
            )
        else:
            raise ValueError(f"Unknown quantizer type: {config.quantizer_type}")

        # Decoder (3D Causal)
        self.decoder = Decoder3D(
            out_channels=patch_channels,
            z_channels=config.z_channels,
            ch=config.ch,
            ch_mult=config.ch_mult,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            dropout=config.dropout,
            temporal_upsample_factors=config.temporal_upsample_factors,
        )

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """
        Encode video to discrete tokens.
        
        Args:
            x: Input video (B, 3, T, H, W), range [-1, 1]
            
        Returns:
            EncoderOutput with latent and indices
        """
        # Patch
        h = self.patcher(x)
        
        # Encode
        z = self.encoder(h)
        
        # Quantize
        z_q, indices, quant_loss, quant_info = self.quantizer(z)
        
        info = {
            "quantizer_loss": quant_loss,
            **quant_info,
        }
        
        return EncoderOutput(latent=z_q, indices=indices, info=info)

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        """
        Decode latent to video.
        
        Args:
            z: Latent tensor (B, z_channels, t, h, w)
            
        Returns:
            DecoderOutput with reconstructed video
        """
        # Decode
        h = self.decoder(z)
        
        # Unpatch
        x_recon = self.unpatcher(h)
        
        return DecoderOutput(reconstruction=x_recon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass: encode and decode.
        
        Args:
            x: Input video
            
        Returns:
            Reconstruction and info dictionary
        """
        enc_out = self.encode(x)
        dec_out = self.decode(enc_out.latent)
        
        info = {**enc_out.info, **dec_out.info}
        
        return dec_out.reconstruction, info

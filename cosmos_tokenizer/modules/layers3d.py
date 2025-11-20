"""3D causal convolutional layers for video tokenization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from cosmos_tokenizer.modules.utils import normalize, nonlinearity


def time2batch(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Reshape video tensor from (B, C, T, H, W) to (B*T, C, H, W).

    Useful for processing temporal frames in parallel as batch.
    """
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return x, T


def batch2time(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Reshape batch tensor from (B*T, C, H, W) to (B, C, T, H, W).

    Inverse of time2batch.
    """
    BT, C, H, W = x.shape
    B = BT // T
    x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x


class CausalConv3d(nn.Module):
    """
    Causal 3D convolution that prevents future information leakage.

    Uses replication padding on temporal axis to ensure only past/current
    frames affect the output at each time step.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: str = "same",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Temporal padding (causal: pad only on the left/past side)
        self.temporal_padding = kernel_size[0] - 1

        # Spatial padding
        if padding == "same":
            self.spatial_padding = (kernel_size[1] // 2, kernel_size[2] // 2)
        else:
            self.spatial_padding = (0, 0)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal temporal padding (replicate on left side only)
        x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0), mode="replicate")

        # Apply spatial padding
        if self.spatial_padding != (0, 0):
            x = F.pad(
                x,
                (
                    self.spatial_padding[1],
                    self.spatial_padding[1],
                    self.spatial_padding[0],
                    self.spatial_padding[0],
                    0,
                    0,
                ),
                mode="replicate",
            )

        return self.conv(x)


class FactorizedConv3d(nn.Module):
    """
    Factorized 3D convolution: separates spatial (1×3×3) and temporal (3×1×1).

    Reduces parameters while maintaining expressiveness. Critical for video at scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal

        # Spatial convolution (1×3×3)
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        # Temporal convolution (3×1×1)
        if causal:
            self.temporal_conv = CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
            )
        else:
            self.temporal_conv = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x


class CausalDownsample3D(nn.Module):
    """Causal 3D downsampling with independent spatial/temporal control."""

    def __init__(
        self,
        in_channels: int,
        spatial_downsample: bool = True,
        temporal_downsample: bool = False,
    ):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample

        # Determine stride
        t_stride = 2 if temporal_downsample else 1
        s_stride = 2 if spatial_downsample else 1

        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 3, 3),
            stride=(t_stride, s_stride, s_stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CausalUpsample3D(nn.Module):
    """Causal 3D upsampling with independent spatial/temporal control."""

    def __init__(
        self,
        in_channels: int,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
    ):
        super().__init__()
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        # Determine scale factors
        self.t_scale = 2.0 if temporal_upsample else 1.0
        self.s_scale = 2.0 if spatial_upsample else 1.0

        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample using nearest neighbor interpolation
        if self.t_scale != 1.0 or self.s_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=(self.t_scale, self.s_scale, self.s_scale),
                mode="nearest",
            )
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    """
    3D ResNet block with causal temporal convolutions.

    Architecture:
        x -> Norm -> SiLU -> Conv3D -> Norm -> SiLU -> Dropout -> Conv3D -> + (skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.causal = causal

        # First conv block
        self.norm1 = normalize(in_channels)
        if causal:
            self.conv1 = FactorizedConv3d(in_channels, self.out_channels, causal=True)
        else:
            self.conv1 = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # Second conv block
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        if causal:
            self.conv2 = FactorizedConv3d(self.out_channels, self.out_channels, causal=True)
        else:
            self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # Skip connection
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        # First conv block
        h = self.norm1(h)
        h = nonlinearity(h, "silu")
        h = self.conv1(h)

        # Second conv block
        h = self.norm2(h)
        h = nonlinearity(h, "silu")
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class CausalAttnBlock3D(nn.Module):
    """
    Causal self-attention block for 3D feature maps.

    Uses triangular masking to prevent attending to future frames.
    """

    def __init__(self, in_channels: int, num_heads: int = 1, causal: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.causal = causal
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.norm = normalize(in_channels)

        # Q, K, V projections (using 1x1x1 conv)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Output projection
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for multi-head attention: (B, C, T, H, W) -> (B, nh, T*H*W, hd)
        B, C, T, H, W = q.shape
        q = q.reshape(B, self.num_heads, self.head_dim, T * H * W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, T * H * W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, T * H * W).transpose(-2, -1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, THW, THW)

        # Apply causal mask
        if self.causal:
            mask = torch.tril(torch.ones(T * H * W, T * H * W, device=attn.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h_ = torch.matmul(attn, v)  # (B, nh, THW, hd)

        # Reshape back
        h_ = h_.transpose(-2, -1).reshape(B, C, T, H, W)
        h_ = self.proj_out(h_)

        return x + h_


class Encoder3D(nn.Module):
    """
    3D Causal Encoder for video.

    Progressive downsampling with causal ResNet blocks and attention.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        z_channels: Number of latent channels
        ch: Base channel count
        ch_mult: Channel multipliers for each resolution level
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply self-attention
        dropout: Dropout probability
        temporal_downsample_factors: Temporal downsampling at each level
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 16,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [32],
        dropout: float = 0.0,
        temporal_downsample_factors: List[bool] = [False, True, True, False],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Initial convolution
        self.conv_in = CausalConv3d(in_channels, ch, kernel_size=(3, 3, 3))

        # Downsampling blocks
        self.down = nn.ModuleList()
        block_in = ch
        curr_res = 256

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            # ResNet blocks
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        causal=True,
                    )
                )
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock3D(block_in, causal=True))

            down = nn.Module()
            down.block = block
            down.attn = attn

            # Downsample
            if i_level != self.num_resolutions - 1:
                temporal_down = temporal_downsample_factors[i_level] if i_level < len(temporal_downsample_factors) else False
                down.downsample = CausalDownsample3D(
                    block_in,
                    spatial_downsample=True,
                    temporal_downsample=temporal_down,
                )
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            causal=True,
        )
        self.mid.attn_1 = CausalAttnBlock3D(block_in, causal=True)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            causal=True,
        )

        # Output
        self.norm_out = normalize(block_in)
        self.conv_out = CausalConv3d(block_in, z_channels, kernel_size=(3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h, "silu")
        h = self.conv_out(h)

        return h


class Decoder3D(nn.Module):
    """
    3D Causal Decoder for video.

    Progressive upsampling with causal ResNet blocks and attention.
    """

    def __init__(
        self,
        out_channels: int = 3,
        z_channels: int = 16,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [32],
        dropout: float = 0.0,
        temporal_upsample_factors: List[bool] = [False, True, True, False],
    ):
        super().__init__()
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # Compute input channels
        block_in = ch * ch_mult[-1]

        # Initial convolution
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=(3, 3, 3))

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            causal=True,
        )
        self.mid.attn_1 = CausalAttnBlock3D(block_in, causal=True)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            causal=True,
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        curr_res = 32

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            # ResNet blocks
            for i_block in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        causal=True,
                    )
                )
                block_in = block_out

                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock3D(block_in, causal=True))

            up = nn.Module()
            up.block = block
            up.attn = attn

            # Upsample
            if i_level != 0:
                temporal_up = temporal_upsample_factors[i_level - 1] if i_level - 1 < len(temporal_upsample_factors) else False
                up.upsample = CausalUpsample3D(
                    block_in,
                    spatial_upsample=True,
                    temporal_upsample=temporal_up,
                )
                curr_res = curr_res * 2

            self.up.insert(0, up)

        # Output
        self.norm_out = normalize(block_in)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=(3, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Initial conv
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h, "silu")
        h = self.conv_out(h)

        return h

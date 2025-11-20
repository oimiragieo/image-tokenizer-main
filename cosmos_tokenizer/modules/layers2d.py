"""2D convolutional layers for image tokenization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

from cosmos_tokenizer.modules.utils import normalize, nonlinearity


class Downsample2D(nn.Module):
    """2D downsampling using strided convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # Stride-2 conv for 2x downsampling
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Asymmetric padding: (left, right, top, bottom)
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    """2D upsampling using repeat-interleave + convolution."""

    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2x upsampling using repeat_interleave
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock2D(nn.Module):
    """
    2D ResNet block with GroupNorm and SiLU activation.

    Architecture:
        x -> Norm -> SiLU -> Conv -> Norm -> SiLU -> Dropout -> Conv -> + (skip)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = conv_shortcut

        # First conv block
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # Time embedding projection (for diffusion models)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.temb_proj = None

        # Second conv block
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        # Skip connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x

        # First conv block
        h = self.norm1(h)
        h = nonlinearity(h, "silu")
        h = self.conv1(h)

        # Add time embedding
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(nonlinearity(temb, "silu"))[:, :, None, None]

        # Second conv block
        h = self.norm2(h)
        h = nonlinearity(h, "silu")
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock2D(nn.Module):
    """
    Self-attention block for 2D feature maps.

    Uses scaled dot-product attention with Q, K, V projections.
    """

    def __init__(self, in_channels: int, num_heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.norm = normalize(in_channels)

        # Q, K, V projections
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Output projection
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape for multi-head attention: (B, C, H, W) -> (B, num_heads, H*W, head_dim)
        B, C, H, W = q.shape
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)  # (B, nh, HW, hd)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)  # (B, nh, HW, hd)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)  # (B, nh, HW, hd)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, HW, HW)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h_ = torch.matmul(attn, v)  # (B, nh, HW, hd)

        # Reshape back
        h_ = h_.transpose(-2, -1).reshape(B, C, H, W)
        h_ = self.proj_out(h_)

        return x + h_


class Encoder2D(nn.Module):
    """
    2D Convolutional Encoder.

    Progressive downsampling with ResNet blocks and self-attention at specific resolutions.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        z_channels: Number of latent channels
        ch: Base channel count
        ch_mult: Channel multipliers for each resolution level
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply self-attention
        dropout: Dropout probability
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList()
        block_in = ch
        curr_res = 256  # Assuming 256x256 input (adjust based on actual input size)

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            # ResNet blocks
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                # Add attention if at specified resolution
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn

            # Downsample (except at last level)
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2D(block_in, with_conv=True)
                curr_res = curr_res // 2

            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock2D(block_in)
        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # Output
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

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


class Decoder2D(nn.Module):
    """
    2D Convolutional Decoder.

    Progressive upsampling with ResNet blocks and self-attention at specific resolutions.

    Args:
        out_channels: Number of output channels (e.g., 3 for RGB)
        z_channels: Number of latent channels
        ch: Base channel count
        ch_mult: Channel multipliers for each resolution level
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply self-attention
        dropout: Dropout probability
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
    ):
        super().__init__()
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions

        # Compute input channels
        block_in = ch * ch_mult[-1]

        # Initial convolution
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock2D(block_in)
        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # Upsampling blocks
        self.up = nn.ModuleList()
        curr_res = 32  # Starting resolution (adjust based on latent size)

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            # ResNet blocks
            for i_block in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock2D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                # Add attention if at specified resolution
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn

            # Upsample (except at first level)
            if i_level != 0:
                up.upsample = Upsample2D(block_in, with_conv=True)
                curr_res = curr_res * 2

            self.up.insert(0, up)  # Insert at beginning for reverse order

        # Output
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

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

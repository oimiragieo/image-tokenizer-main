"""
Patching modules for efficient compression before encoding.

Implements two patching strategies:
1. Haar wavelet transform (lossless, preserves high-frequency details)
2. Rearrange patching (simple spatial-to-channel conversion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple


class HaarPatch2D(nn.Module):
    """
    2D Haar wavelet patching.

    Decomposes image into LL, LH, HL, HH components:
    - LL: Low-frequency (approximation)
    - LH: Horizontal edges
    - HL: Vertical edges
    - HH: Diagonal edges

    Reduces spatial resolution by 2× while preserving all information in channels.
    """

    def __init__(self, patch_size: int = 2):
        super().__init__()
        assert patch_size == 2, "Haar patching only supports patch_size=2"
        self.patch_size = patch_size

        # Haar wavelet kernels
        # Low-pass: average
        # High-pass: difference
        self.register_buffer(
            "haar_low",
            torch.tensor([1.0, 1.0]) / 2.0,
        )
        self.register_buffer(
            "haar_high",
            torch.tensor([1.0, -1.0]) / 2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patched tensor (B, 4*C, H//2, W//2)
        """
        B, C, H, W = x.shape

        # Ensure dimensions are even
        assert H % 2 == 0 and W % 2 == 0, f"Height and width must be even, got {H}x{W}"

        # Apply 1D Haar transform on rows
        x_low_rows = F.conv2d(
            x,
            self.haar_low.view(1, 1, 1, 2).repeat(C, 1, 1, 1),
            stride=(1, 2),
            groups=C,
        )
        x_high_rows = F.conv2d(
            x,
            self.haar_high.view(1, 1, 1, 2).repeat(C, 1, 1, 1),
            stride=(1, 2),
            groups=C,
        )

        # Apply 1D Haar transform on columns
        LL = F.conv2d(
            x_low_rows,
            self.haar_low.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
            stride=(2, 1),
            groups=C,
        )
        LH = F.conv2d(
            x_low_rows,
            self.haar_high.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
            stride=(2, 1),
            groups=C,
        )
        HL = F.conv2d(
            x_high_rows,
            self.haar_low.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
            stride=(2, 1),
            groups=C,
        )
        HH = F.conv2d(
            x_high_rows,
            self.haar_high.view(1, 1, 2, 1).repeat(C, 1, 1, 1),
            stride=(2, 1),
            groups=C,
        )

        # Concatenate along channel dimension
        out = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4*C, H//2, W//2)

        return out


class HaarUnpatch2D(nn.Module):
    """
    2D Haar wavelet unpatching (inverse transform).

    Reconstructs image from LL, LH, HL, HH components.
    """

    def __init__(self, patch_size: int = 2):
        super().__init__()
        assert patch_size == 2, "Haar unpatching only supports patch_size=2"
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patched tensor (B, 4*C, H, W)

        Returns:
            Reconstructed tensor (B, C, 2*H, 2*W)
        """
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Channels must be divisible by 4"
        C = C4 // 4

        # Split into LL, LH, HL, HH
        LL, LH, HL, HH = torch.chunk(x, 4, dim=1)

        # Inverse Haar transform
        # Reconstruct rows from column components
        low_rows = (LL + LH) * 2  # Low-frequency rows
        high_rows = (HL + HH) * 2  # High-frequency rows

        # Upsample rows (interleave)
        rows = torch.zeros(B, C, 2 * H, W, device=x.device, dtype=x.dtype)
        rows[:, :, 0::2, :] = low_rows
        rows[:, :, 1::2, :] = low_rows

        rows_h = torch.zeros(B, C, 2 * H, W, device=x.device, dtype=x.dtype)
        rows_h[:, :, 0::2, :] = high_rows
        rows_h[:, :, 1::2, :] = -high_rows

        # Reconstruct columns
        low_cols = (LL + HL) * 2
        high_cols = (LH + HH) * 2

        # Alternative reconstruction using proper inverse
        # Upsample and reconstruct
        out = torch.zeros(B, C, 2 * H, 2 * W, device=x.device, dtype=x.dtype)

        # LL contributes to all positions
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH)
        out[:, :, 0::2, 1::2] = (LL - LH + HL - HH)
        out[:, :, 1::2, 0::2] = (LL + LH - HL - HH)
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH)

        return out


class HaarPatch3D(nn.Module):
    """
    3D Haar wavelet patching for video.

    Produces 8 components (2^3) from temporal and spatial decomposition.
    """

    def __init__(self, patch_size: int = 2, patch_temporal: bool = True):
        super().__init__()
        assert patch_size == 2, "Haar patching only supports patch_size=2"
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal

        self.register_buffer("haar_low", torch.tensor([1.0, 1.0]) / 2.0)
        self.register_buffer("haar_high", torch.tensor([1.0, -1.0]) / 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)

        Returns:
            Patched tensor (B, 8*C, T//2, H//2, W//2) if patch_temporal
            else (B, 4*C, T, H//2, W//2)
        """
        B, C, T, H, W = x.shape

        # Apply 2D Haar on spatial dimensions
        # Process each frame separately using time2batch
        from cosmos_tokenizer.modules.layers3d import time2batch, batch2time

        x_2d, T_orig = time2batch(x)  # (B*T, C, H, W)

        # Apply spatial Haar patching
        haar_2d = HaarPatch2D(patch_size=2)
        x_patched_2d = haar_2d(x_2d)  # (B*T, 4*C, H//2, W//2)

        # Convert back to video
        x_patched = batch2time(x_patched_2d, T_orig)  # (B, 4*C, T, H//2, W//2)

        if not self.patch_temporal or T == 1:
            return x_patched

        # Apply 1D Haar on temporal dimension
        assert T % 2 == 0, f"Temporal dimension must be even, got {T}"

        x_patched = x_patched.permute(0, 2, 1, 3, 4)  # (B, T, 4*C, H//2, W//2)
        B, T, C4, H2, W2 = x_patched.shape

        # Temporal low and high pass
        x_low_t = F.conv3d(
            x_patched.reshape(B, 1, T, C4 * H2, W2),
            self.haar_low.view(1, 1, 2, 1, 1),
            stride=(2, 1, 1),
        ).reshape(B, T // 2, C4, H2, W2)

        x_high_t = F.conv3d(
            x_patched.reshape(B, 1, T, C4 * H2, W2),
            self.haar_high.view(1, 1, 2, 1, 1),
            stride=(2, 1, 1),
        ).reshape(B, T // 2, C4, H2, W2)

        # Concatenate temporal components
        out = torch.cat([x_low_t, x_high_t], dim=2)  # (B, T//2, 8*C, H//2, W//2)
        out = out.permute(0, 2, 1, 3, 4)  # (B, 8*C, T//2, H//2, W//2)

        return out


class HaarUnpatch3D(nn.Module):
    """3D Haar wavelet unpatching (inverse transform)."""

    def __init__(self, patch_size: int = 2, patch_temporal: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patched tensor (B, 8*C, T, H, W) or (B, 4*C, T, H, W)

        Returns:
            Reconstructed tensor (B, C, 2*T, 2*H, 2*W) or (B, C, T, 2*H, 2*W)
        """
        B, Cx, T, H, W = x.shape

        if self.patch_temporal and Cx % 8 == 0:
            C = Cx // 8
            # Split temporal components
            x_low_t, x_high_t = torch.chunk(x, 2, dim=1)  # Each (B, 4*C, T, H, W)

            # Reconstruct temporal dimension
            x_patched = torch.zeros(B, 4 * C, 2 * T, H, W, device=x.device, dtype=x.dtype)
            x_patched[:, :, 0::2, :, :] = x_low_t + x_high_t
            x_patched[:, :, 1::2, :, :] = x_low_t - x_high_t
        else:
            assert Cx % 4 == 0, "Channels must be divisible by 4"
            C = Cx // 4
            x_patched = x

        # Apply 2D Haar unpatching on spatial dimensions
        from cosmos_tokenizer.modules.layers3d import time2batch, batch2time

        x_2d, T_new = time2batch(x_patched)  # (B*T, 4*C, H, W)

        # Apply spatial Haar unpatching
        haar_2d = HaarUnpatch2D(patch_size=2)
        x_unpatch_2d = haar_2d(x_2d)  # (B*T, C, 2*H, 2*W)

        # Convert back to video
        out = batch2time(x_unpatch_2d, T_new)  # (B, C, T, 2*H, 2*W)

        return out


class RearrangePatch(nn.Module):
    """
    Rearrange patching using einops.

    Simpler alternative to Haar - just rearranges spatial patches to channels.
    """

    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W) or (B, C, T, H, W)

        Returns:
            Patched tensor with patches moved to channels
        """
        if x.ndim == 4:
            # 2D image
            p = self.patch_size
            return rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=p, p2=p)
        else:
            # 3D video
            p = self.patch_size
            return rearrange(x, "b c t (h p1) (w p2) -> b (c p1 p2) t h w", p1=p, p2=p)


class RearrangeUnpatch(nn.Module):
    """Inverse of RearrangePatch."""

    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patched tensor

        Returns:
            Unpatched tensor
        """
        if x.ndim == 4:
            # 2D image
            p = self.patch_size
            return rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=p, p2=p)
        else:
            # 3D video
            p = self.patch_size
            return rearrange(x, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", p1=p, p2=p)

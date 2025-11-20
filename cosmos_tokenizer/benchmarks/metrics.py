"""
Reconstruction quality metrics for tokenizers.

Provides:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- MSE (Mean Squared Error)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class QualityMetrics:
    """
    Comprehensive quality metrics for reconstruction.

    Attributes:
        psnr: Peak Signal-to-Noise Ratio (dB)
        ssim: Structural Similarity Index (0-1)
        mse: Mean Squared Error
        lpips: Learned Perceptual Image Patch Similarity (if available)
    """

    psnr: float
    ssim: float
    mse: float
    lpips: Optional[float] = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mse": self.mse,
            "lpips": self.lpips,
        }

    def __str__(self):
        """Human-readable string representation."""
        s = f"PSNR: {self.psnr:.2f}dB, SSIM: {self.ssim:.4f}, MSE: {self.mse:.6f}"
        if self.lpips is not None:
            s += f", LPIPS: {self.lpips:.4f}"
        return s


def calculate_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor

    Returns:
        MSE value
    """
    return F.mse_loss(original, reconstructed).item()


def calculate_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        original: Original tensor (B, C, H, W) or (B, C, T, H, W)
        reconstructed: Reconstructed tensor
        max_value: Maximum possible value (1.0 for normalized [-1,1])

    Returns:
        PSNR in dB
    """
    mse = calculate_mse(original, reconstructed)

    if mse == 0:
        return float("inf")

    # For data in range [-1, 1], max_value is 2 (range from -1 to 1)
    # But conventionally we use 1.0 for normalized data
    psnr = 20 * torch.log10(torch.tensor(max_value * 2)) - 10 * torch.log10(
        torch.tensor(mse)
    )

    return psnr.item()


def calculate_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> float:
    """
    Calculate Structural Similarity Index (SSIM).

    This is a simplified implementation. For production use, consider
    using torchmetrics.StructuralSimilarityIndexMeasure or pytorch-msssim.

    Args:
        original: Original tensor (B, C, H, W)
        reconstructed: Reconstructed tensor
        window_size: Window size for SSIM calculation
        size_average: Average over batch and channels

    Returns:
        SSIM value (0-1, higher is better)
    """
    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor(
        [
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma**2)))
            for x in range(window_size)
        ]
    )
    window_1d = (gauss / gauss.sum()).unsqueeze(1)
    window = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(original.device)

    # Expand window for all channels
    num_channels = original.size(1)
    window = window.expand(num_channels, 1, window_size, window_size).contiguous()

    # Calculate statistics
    mu1 = F.conv2d(original, window, padding=window_size // 2, groups=num_channels)
    mu2 = F.conv2d(
        reconstructed, window, padding=window_size // 2, groups=num_channels
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(original * original, window, padding=window_size // 2, groups=num_channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            reconstructed * reconstructed,
            window,
            padding=window_size // 2,
            groups=num_channels,
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            original * reconstructed,
            window,
            padding=window_size // 2,
            groups=num_channels,
        )
        - mu1_mu2
    )

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3]).item()


def calculate_lpips(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    lpips_model=None,
) -> Optional[float]:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Requires lpips package: pip install lpips

    Args:
        original: Original tensor (B, C, H, W)
        reconstructed: Reconstructed tensor
        lpips_model: Pre-loaded LPIPS model (optional)

    Returns:
        LPIPS value (lower is better), or None if lpips not available
    """
    try:
        import lpips

        if lpips_model is None:
            lpips_model = lpips.LPIPS(net="alex").to(original.device)

        with torch.no_grad():
            lpips_value = lpips_model(original, reconstructed)

        return lpips_value.mean().item()

    except ImportError:
        return None


def compute_all_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    use_lpips: bool = False,
) -> QualityMetrics:
    """
    Compute all reconstruction quality metrics.

    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
        use_lpips: Whether to compute LPIPS (requires lpips package)

    Returns:
        QualityMetrics with all computed metrics
    """
    mse = calculate_mse(original, reconstructed)
    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim(original, reconstructed)

    lpips_value = None
    if use_lpips:
        lpips_value = calculate_lpips(original, reconstructed)

    return QualityMetrics(
        psnr=psnr,
        ssim=ssim,
        mse=mse,
        lpips=lpips_value,
    )

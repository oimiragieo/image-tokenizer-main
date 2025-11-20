"""Utility functions for modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def normalize(in_channels: int, num_groups: int = 32, eps: float = 1e-6) -> nn.GroupNorm:
    """Create GroupNorm layer with automatic group adjustment."""
    num_groups = min(num_groups, in_channels)
    while in_channels % num_groups != 0:
        num_groups = num_groups // 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)


def nonlinearity(x: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """Apply nonlinearity activation function."""
    if activation == "silu" or activation == "swish":
        return F.silu(x)
    elif activation == "relu":
        return F.relu(x)
    elif activation == "gelu":
        return F.gelu(x)
    elif activation == "tanh":
        return torch.tanh(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def log(t: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Numerically stable logarithm."""
    return torch.log(t.clamp(min=eps))


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator for gradients."""
    return z + (z.round() - z).detach()


def exists(val) -> bool:
    """Check if value exists (is not None)."""
    return val is not None


def default(val, d):
    """Return val if it exists, otherwise return d."""
    return val if exists(val) else d


def l2norm(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 normalization along dimension."""
    return F.normalize(t, dim=dim, p=2)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    """Exponential moving average update in-place."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x: torch.Tensor, n_categories: int, epsilon: float = 1e-5):
    """Apply Laplace smoothing to probability distribution."""
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int) -> torch.Tensor:
    """Create uniformly initialized tensor."""
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel noise."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits: torch.Tensor, temperature: float = 1.0, dim: int = -1, hard: bool = False
) -> torch.Tensor:
    """Gumbel softmax sampling."""
    gumbel_noise_t = gumbel_noise(logits)
    y = (logits + gumbel_noise_t) / temperature
    y_soft = F.softmax(y, dim=dim)

    if hard:
        # Straight through estimator
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Sample random vectors from batch."""
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def batched_embedding(indices: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
    """Efficient batched embedding lookup."""
    batch_size, seq_len = indices.shape[:2]
    indices_flat = indices.reshape(-1)
    embeds_flat = embeds[indices_flat]
    return embeds_flat.reshape(batch_size, seq_len, -1)


def compute_distances(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Euclidean distances between x and codebook vectors.

    Args:
        x: Input tensor of shape (B, D) or (B, H, W, D)
        codebook: Codebook tensor of shape (K, D)

    Returns:
        Distances of shape (B, K) or (B, H, W, K)
    """
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x · c
    x_sq = torch.sum(x**2, dim=-1, keepdim=True)  # (B, ..., 1)
    c_sq = torch.sum(codebook**2, dim=-1, keepdim=False)  # (K,)
    dot_product = torch.matmul(x, codebook.T)  # (B, ..., K)
    distances = x_sq + c_sq - 2 * dot_product
    return distances


def orthogonal_loss_fn(t: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal regularization loss for codebook.
    Encourages codebook vectors to be orthogonal.
    """
    # Normalize
    n = F.normalize(t, dim=-1)
    # Compute gram matrix
    gram = torch.matmul(n, n.T)
    # Penalize off-diagonal elements
    mask = torch.eye(gram.shape[0], device=gram.device, dtype=torch.bool)
    loss = (gram.masked_fill(mask, 0.0) ** 2).sum()
    return loss

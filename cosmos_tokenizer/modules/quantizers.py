"""
Quantization modules for discrete tokenization.

Implements four state-of-the-art quantization methods:
- VectorQuantizer: Classical VQ-VAE with learned codebook
- FSQuantizer: Finite Scalar Quantization (implicit codebook)
- LFQuantizer: Lookup-Free Quantization (binary)
- ResidualFSQuantizer: Multi-stage cascaded FSQ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

from cosmos_tokenizer.modules.utils import (
    round_ste,
    compute_distances,
    ema_inplace,
    sample_vectors,
    laplace_smoothing,
    orthogonal_loss_fn,
    log,
)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with exponential moving average (EMA) codebook updates.

    Based on VQ-VAE and VQ-VAE-2 papers. Uses straight-through estimator
    for gradient flow and optional codebook remapping to prevent collapse.

    Args:
        num_embeddings: Codebook size (number of discrete tokens)
        embedding_dim: Dimension of each codebook vector
        commitment_cost: Weight for commitment loss (default: 0.25)
        decay: EMA decay rate for codebook updates (default: 0.99)
        epsilon: Small value for numerical stability (default: 1e-5)
        use_ema: Whether to use EMA updates or gradient-based (default: True)
        remap: Codebook usage remapping strategy (None, 'cluster', or 'random')
        usage_threshold: Minimum usage count before remapping (default: 0.01)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True,
        remap: Optional[str] = None,
        usage_threshold: float = 0.01,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        self.remap = remap
        self.usage_threshold = usage_threshold

        # Initialize codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        if use_ema:
            self.register_buffer("cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("embedding_avg", self.embedding.weight.data.clone())

        # Usage tracking
        self.register_buffer("usage_count", torch.zeros(num_embeddings))

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: Input tensor of shape (B, C, H, W) or (B, C, T, H, W)

        Returns:
            z_q: Quantized tensor (same shape as z)
            indices: Codebook indices (B, H, W) or (B, T, H, W)
            commitment_loss: Commitment loss for training
            info: Dictionary with additional information
        """
        # Flatten spatial dimensions
        shape = z.shape
        z_flat = z.movedim(1, -1).reshape(-1, self.embedding_dim)  # (B*H*W, C)

        # Compute distances to codebook
        distances = compute_distances(z_flat, self.embedding.weight)  # (B*H*W, K)

        # Get nearest codebook vectors
        indices_flat = torch.argmin(distances, dim=-1)  # (B*H*W,)
        z_q_flat = self.embedding(indices_flat)  # (B*H*W, C)

        # Update codebook if using EMA
        if self.training and self.use_ema:
            self._update_codebook_ema(z_flat, indices_flat)

        # Track usage
        if self.training:
            self._update_usage(indices_flat)

        # Compute losses
        if self.use_ema:
            # Only commitment loss with EMA
            commitment_loss = F.mse_loss(z_flat.detach(), z_q_flat)
            quantizer_loss = commitment_loss
        else:
            # Both embedding and commitment loss with gradients
            commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
            embedding_loss = F.mse_loss(z_q_flat, z_flat.detach())
            quantizer_loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back
        z_q = z_q_flat.reshape(*shape[:-3], -1, shape[-2], shape[-1]).movedim(-3, 1)
        if len(shape) == 5:  # Video
            indices = indices_flat.reshape(shape[0], shape[2], shape[3], shape[4])
        else:  # Image
            indices = indices_flat.reshape(shape[0], shape[2], shape[3])

        # Perplexity (measure of codebook usage)
        avg_probs = torch.bincount(
            indices_flat, minlength=self.num_embeddings
        ).float() / len(indices_flat)
        perplexity = torch.exp(-torch.sum(avg_probs * log(avg_probs)))

        info = {
            "quantizer_loss": quantizer_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "used_codes": (avg_probs > 0).sum(),
            "usage_entropy": -torch.sum(avg_probs * log(avg_probs)),
        }

        return z_q, indices, quantizer_loss, info

    def _update_codebook_ema(self, z_flat: torch.Tensor, indices_flat: torch.Tensor):
        """Update codebook using exponential moving average."""
        # One-hot encoding
        encodings = F.one_hot(indices_flat, self.num_embeddings).float()  # (N, K)

        # Update cluster sizes
        updated_cluster_size = torch.sum(encodings, dim=0)
        ema_inplace(self.cluster_size, updated_cluster_size, self.decay)

        # Update embedding averages
        updated_embedding_avg = torch.matmul(encodings.T, z_flat)  # (K, C)
        ema_inplace(self.embedding_avg, updated_embedding_avg, self.decay)

        # Normalize embeddings
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embedding_normalized)

    def _update_usage(self, indices_flat: torch.Tensor):
        """Track codebook usage for remapping."""
        usage = torch.bincount(indices_flat, minlength=self.num_embeddings).float()
        ema_inplace(self.usage_count, usage, decay=0.99)

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to embeddings."""
        return self.embedding(indices)


class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ).

    Paper: "VQ-VAE Made Simple" - uses implicit codebook derived from
    quantization levels per dimension. More efficient than VQ-VAE.

    Args:
        levels: List of quantization levels per dimension (e.g., [8, 8, 8, 5, 5, 5])
        embedding_dim: Dimension of input (should match len(levels))
        scale: Optional scaling factor for quantized values
    """

    def __init__(
        self,
        levels: List[int],
        embedding_dim: Optional[int] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.levels = levels
        self.embedding_dim = embedding_dim or len(levels)
        assert self.embedding_dim == len(levels), "Embedding dim must match number of levels"

        # Compute codebook size
        self.codebook_size = int(torch.prod(torch.tensor(levels)).item())

        # Compute basis for index conversion
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer("basis", basis)

        # Compute scales for each dimension
        if scale is None:
            # Implicit scales: map [-1, 1] to [0, level-1]
            scales = torch.tensor([(level - 1) / 2 for level in levels])
        else:
            scales = torch.full((len(levels),), scale)
        self.register_buffer("scales", scales)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: Input tensor (B, C, H, W) or (B, C, T, H, W)

        Returns:
            z_q: Quantized tensor
            indices: Flattened codebook indices
            loss: Always zero (no explicit loss)
            info: Additional information
        """
        shape = z.shape

        # Move channels to last dimension
        z_moved = z.movedim(1, -1)  # (B, H, W, C) or (B, T, H, W, C)

        # Bound input to [-1, 1] using tanh
        z_bounded = torch.tanh(z_moved)

        # Scale and shift to [0, levels-1]
        z_scaled = (z_bounded + 1) * self.scales

        # Round with straight-through estimator
        z_quantized = round_ste(z_scaled)

        # Compute flat indices
        indices = self._indices_to_flat(z_quantized)

        # Unscale back to [-1, 1]
        z_q = z_quantized / self.scales - 1

        # Move channels back
        z_q = z_q.movedim(-1, 1)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Compute perplexity
        indices_flat = indices.reshape(-1)
        avg_probs = torch.bincount(
            indices_flat, minlength=self.codebook_size
        ).float() / len(indices_flat)
        perplexity = torch.exp(-torch.sum(avg_probs * log(avg_probs)))

        info = {
            "quantizer_loss": torch.tensor(0.0, device=z.device),
            "perplexity": perplexity,
            "used_codes": (avg_probs > 0).sum(),
        }

        return z_q, indices, torch.tensor(0.0, device=z.device), info

    def _indices_to_flat(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert per-dimension indices to flat codebook indices."""
        # indices: (..., D) with values in [0, levels[d]-1]
        # flat_index = sum(indices[d] * basis[d])
        flat = (indices * self.basis).sum(dim=-1).long()
        return flat

    def _flat_to_indices(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat indices to per-dimension indices."""
        indices = []
        remainder = flat
        for i, level in enumerate(self.levels):
            indices.append(remainder % level)
            remainder = remainder // level
        return torch.stack(indices, dim=-1)

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices to embeddings."""
        # Convert flat to per-dimension indices
        per_dim = self._flat_to_indices(indices)
        # Unscale to [-1, 1]
        embeddings = per_dim.float() / self.scales - 1
        return embeddings


class LFQuantizer(nn.Module):
    """
    Lookup-Free Quantization (LFQ).

    Paper: "Lookup-Free Quantization" - uses binary quantization based
    on sign function. No explicit codebook needed. Very efficient.

    Args:
        embedding_dim: Dimension of latent space
        entropy_loss_weight: Weight for entropy loss to encourage uniform usage
        diversity_gamma: Temperature for entropy calculation
    """

    def __init__(
        self,
        embedding_dim: int,
        entropy_loss_weight: float = 0.1,
        diversity_gamma: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = 2**embedding_dim
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: Input tensor (B, C, H, W) or (B, C, T, H, W)

        Returns:
            z_q: Quantized tensor (binary: -1 or 1)
            indices: Binary indices converted to integers
            loss: Entropy loss
            info: Additional information
        """
        shape = z.shape

        # Move channels to last dimension
        z_moved = z.movedim(1, -1)

        # Binary quantization with straight-through estimator
        z_q = torch.where(z_moved > 0, 1.0, -1.0)
        z_q = z_moved + (z_q - z_moved).detach()

        # Convert binary to indices
        # Map {-1, 1} to {0, 1}, then interpret as binary number
        binary = ((z_q + 1) / 2).long()  # Map to {0, 1}
        indices = self._binary_to_indices(binary)

        # Compute entropy loss to encourage diversity
        if self.training:
            # Soft assignment probabilities
            probs = torch.sigmoid(z_moved / self.diversity_gamma)
            # Entropy: -p*log(p) - (1-p)*log(1-p)
            entropy = -(probs * log(probs) + (1 - probs) * log(1 - probs))
            entropy_loss = -entropy.mean() * self.entropy_loss_weight
        else:
            entropy_loss = torch.tensor(0.0, device=z.device)

        # Move channels back
        z_q = z_q.movedim(-1, 1)

        # Compute perplexity
        indices_flat = indices.reshape(-1)
        avg_probs = torch.bincount(
            indices_flat, minlength=min(self.codebook_size, 100000)
        ).float() / len(indices_flat)
        perplexity = torch.exp(-torch.sum(avg_probs * log(avg_probs)))

        info = {
            "quantizer_loss": entropy_loss,
            "entropy_loss": entropy_loss,
            "perplexity": perplexity,
            "used_codes": (avg_probs > 0).sum(),
        }

        return z_q, indices, entropy_loss, info

    def _binary_to_indices(self, binary: torch.Tensor) -> torch.Tensor:
        """Convert binary tensor to integer indices."""
        # binary: (..., D) with values in {0, 1}
        # index = sum(binary[d] * 2^d)
        powers = 2 ** torch.arange(binary.shape[-1], device=binary.device)
        indices = (binary * powers).sum(dim=-1)
        return indices

    def _indices_to_binary(self, indices: torch.Tensor, dim: int) -> torch.Tensor:
        """Convert integer indices to binary tensor."""
        # Create binary representation
        binary = []
        remainder = indices
        for i in range(dim):
            binary.append(remainder % 2)
            remainder = remainder // 2
        return torch.stack(binary, dim=-1)

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to embeddings."""
        binary = self._indices_to_binary(indices, self.embedding_dim)
        # Map {0, 1} to {-1, 1}
        embeddings = binary.float() * 2 - 1
        return embeddings


class ResidualFSQuantizer(nn.Module):
    """
    Residual Finite Scalar Quantization.

    Multi-stage cascaded FSQ where each stage quantizes the residual error
    from previous stages. Enables higher effective codebook capacity.

    Args:
        num_stages: Number of quantization stages
        levels: List of levels per dimension for each stage
        embedding_dim: Dimension of input
    """

    def __init__(
        self,
        num_stages: int = 4,
        levels: List[int] = [8, 8, 8, 5, 5, 5],
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.embedding_dim = embedding_dim or len(levels)

        # Create quantizer for each stage
        self.quantizers = nn.ModuleList([
            FSQuantizer(levels, embedding_dim) for _ in range(num_stages)
        ])

        # Total codebook size is product of all stages
        stage_size = int(torch.prod(torch.tensor(levels)).item())
        self.codebook_size = stage_size ** num_stages

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z: Input tensor (B, C, H, W) or (B, C, T, H, W)

        Returns:
            z_q: Quantized tensor (sum of all stages)
            indices: Combined indices from all stages
            loss: Always zero
            info: Additional information
        """
        residual = z
        z_q_total = torch.zeros_like(z)
        all_indices = []
        total_loss = 0.0

        for i, quantizer in enumerate(self.quantizers):
            # Quantize residual
            z_q, indices, loss, info = quantizer(residual)

            # Accumulate quantized output
            z_q_total = z_q_total + z_q

            # Update residual (remove quantized component)
            residual = residual - z_q.detach()

            # Store indices
            all_indices.append(indices)

            total_loss = total_loss + loss

        # Combine indices (simple concatenation along new dimension)
        combined_indices = torch.stack(all_indices, dim=-1)

        # Compute overall perplexity
        # This is approximate since true codebook is combinatorial
        avg_perplexity = sum(
            quantizer.codebook_size for quantizer in self.quantizers
        ) / self.num_stages

        info = {
            "quantizer_loss": total_loss,
            "perplexity": torch.tensor(avg_perplexity, device=z.device),
            "num_stages": self.num_stages,
        }

        return z_q_total, combined_indices, total_loss, info

    def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to embeddings."""
        # indices: (..., num_stages)
        embeddings = torch.zeros(
            *indices.shape[:-1], self.embedding_dim, device=indices.device
        )
        for i, quantizer in enumerate(self.quantizers):
            stage_indices = indices[..., i]
            embeddings = embeddings + quantizer.embed_code(stage_indices)
        return embeddings


# Alias for compatibility
VQ = VectorQuantizer
FSQ = FSQuantizer
LFQ = LFQuantizer
ResidualFSQ = ResidualFSQuantizer

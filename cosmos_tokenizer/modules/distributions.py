"""Probability distributions for continuous tokenization (VAE-style)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class DiagonalGaussianDistribution:
    """
    Diagonal (factorized) Gaussian distribution for VAE-style continuous tokenization.

    Assumes independence between dimensions with diagonal covariance matrix.
    Uses reparameterization trick for differentiable sampling.

    Args:
        parameters: Tensor containing concatenated mean and log-variance
        deterministic: If True, use mean without sampling
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        # Clamp logvar for numerical stability
        self.logvar = torch.clamp(self.logvar, min=-30.0, max=20.0)

        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        """Sample from distribution using reparameterization trick."""
        if self.deterministic:
            return self.mean
        else:
            # Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
            eps = torch.randn_like(self.mean)
            return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """
        Compute KL divergence.

        If other is None, compute KL(q||N(0,1)) (standard normal prior).
        Otherwise compute KL(self||other).
        """
        if self.deterministic:
            return torch.tensor(0.0, device=self.mean.device)

        if other is None:
            # KL divergence with standard normal N(0,1)
            # KL(q||p) = 0.5 * sum(1 + log(σ²) - μ² - σ²)
            kl = -0.5 * torch.sum(1 + self.logvar - self.mean.pow(2) - self.var, dim=[1, 2, 3])
            return kl.mean()
        else:
            # KL divergence between two Gaussians
            # KL(q||p) = 0.5 * sum(log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1)
            kl = 0.5 * torch.sum(
                other.logvar - self.logvar
                + (self.var + (self.mean - other.mean).pow(2)) / other.var
                - 1.0,
                dim=[1, 2, 3]
            )
            return kl.mean()

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
        """
        Compute negative log-likelihood of a sample.

        NLL = -log p(x|z) = 0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
        """
        if self.deterministic:
            return torch.tensor(0.0, device=sample.device)

        logtwopi = np.log(2.0 * np.pi)
        nll = 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=dims
        )
        return nll.mean()

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the distribution.

        H = 0.5 * (n * log(2πe) + sum(log(σ²)))
        """
        if self.deterministic:
            return torch.tensor(0.0, device=self.mean.device)

        n_dims = np.prod(self.mean.shape[1:])
        log2pie = np.log(2.0 * np.pi * np.e)
        entropy = 0.5 * (n_dims * log2pie + torch.sum(self.logvar, dim=[1, 2, 3]))
        return entropy.mean()


class AutoencoderKL(nn.Module):
    """
    Wrapper for continuous autoencoder with KL regularization.

    Combines encoder output projection to mean/logvar and distribution sampling.
    """

    def __init__(
        self,
        z_channels: int,
        double_z: bool = True,
        kl_weight: float = 1e-6,
    ):
        """
        Args:
            z_channels: Number of latent channels
            double_z: If True, encoder outputs 2*z_channels (mean + logvar)
            kl_weight: Weight for KL divergence loss
        """
        super().__init__()
        self.z_channels = z_channels
        self.double_z = double_z
        self.kl_weight = kl_weight

        # Output channels (doubled for mean and logvar)
        self.quant_channels = 2 * z_channels if double_z else z_channels

    def encode(self, h: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Encode to latent distribution and sample.

        Args:
            h: Encoder output (B, quant_channels, H, W)
            deterministic: If True, use mean without sampling

        Returns:
            z: Sampled latent (B, z_channels, H, W)
            info: Dictionary with posterior and KL divergence
        """
        posterior = DiagonalGaussianDistribution(h, deterministic=deterministic)
        z = posterior.sample()

        # Compute KL divergence
        kl_loss = posterior.kl() * self.kl_weight if not deterministic else 0.0

        info = {
            "posterior": posterior,
            "kl_loss": kl_loss,
        }

        return z, info

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent. Just pass through (decoding handled by main decoder).

        Args:
            z: Latent tensor (B, z_channels, H, W)

        Returns:
            z: Same tensor
        """
        return z

    def forward(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full forward pass: encode and optionally decode.

        Args:
            h: Encoder output
            deterministic: If True, use mean without sampling

        Returns:
            z: Sampled latent
            info: Dictionary with distribution info
        """
        return self.encode(h, deterministic=deterministic)

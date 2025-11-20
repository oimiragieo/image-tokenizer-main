"""Loss functions for training tokenizer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for image/video tokenizers.
    
    Supports L1, L2, and perceptual losses.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            prediction: Predicted image/video
            target: Ground truth
            
        Returns:
            Loss value
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(prediction, target, reduction=self.reduction)
        elif self.loss_type == "l2" or self.loss_type == "mse":
            loss = F.mse_loss(prediction, target, reduction=self.reduction)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(prediction, target, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    
    Compares feature representations at multiple layers.
    """

    def __init__(
        self,
        layers: list = [2, 7, 12, 21, 30],
        weights: Optional[list] = None,
    ):
        super().__init__()
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            # Fallback for older torchvision
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
        
        self.layers = layers
        self.weights = weights or [1.0] * len(layers)
        
        # Extract feature extractors
        self.feature_extractors = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.feature_extractors.append(vgg[prev_layer:layer + 1])
            prev_layer = layer + 1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            prediction: Predicted image (B, 3, H, W), range [-1, 1]
            target: Ground truth image
            
        Returns:
            Loss value
        """
        # Normalize from [-1, 1] to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=prediction.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=prediction.device)
        
        pred_norm = (prediction * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
        pred_norm = (pred_norm - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        target_norm = (target * 0.5 + 0.5)
        target_norm = (target_norm - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        
        # Extract features and compute loss
        total_loss = 0.0
        pred_features = pred_norm
        target_features = target_norm
        
        for i, extractor in enumerate(self.feature_extractors):
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            
            loss = F.mse_loss(pred_features, target_features)
            total_loss += self.weights[i] * loss
        
        return total_loss


class TokenizerLoss(nn.Module):
    """
    Combined loss for tokenizer training.
    
    Includes reconstruction, perceptual, quantizer, and KL losses.
    """

    def __init__(
        self,
        recon_loss_type: str = "l1",
        perceptual_weight: float = 1.0,
        quantizer_weight: float = 1.0,
        kl_weight: float = 1e-6,
        use_perceptual: bool = True,
    ):
        super().__init__()
        
        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type)
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
        
        self.perceptual_weight = perceptual_weight
        self.quantizer_weight = quantizer_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        info: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            prediction: Predicted image/video
            target: Ground truth
            info: Dictionary with additional info (quantizer_loss, kl_loss, etc.)
            
        Returns:
            Dictionary with individual and total losses
        """
        # Reconstruction loss
        recon_loss = self.recon_loss(prediction, target)
        
        # Perceptual loss
        if self.perceptual_loss is not None and prediction.ndim == 4:
            # Only for images (not videos)
            percep_loss = self.perceptual_loss(prediction, target)
        else:
            percep_loss = torch.tensor(0.0, device=prediction.device)
        
        # Quantizer loss
        quantizer_loss = info.get("quantizer_loss", torch.tensor(0.0, device=prediction.device))
        
        # KL loss
        kl_loss = info.get("kl_loss", torch.tensor(0.0, device=prediction.device))
        
        # Total loss
        total_loss = (
            recon_loss
            + self.perceptual_weight * percep_loss
            + self.quantizer_weight * quantizer_loss
            + self.kl_weight * kl_loss
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": percep_loss,
            "quantizer_loss": quantizer_loss,
            "kl_loss": kl_loss,
        }

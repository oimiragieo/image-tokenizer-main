"""
Token counting module for multimodal tokenization with billing integration.

Provides:
- Token counting across modalities (image, video, audio, text)
- Cost calculation based on token usage
- Usage aggregation and reporting
- Integration with billing systems
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import torch

from cosmos_tokenizer.core.modality import Modality


@dataclass
class TokenUsage:
    """
    Token usage record for a single operation.

    Attributes:
        modality: Modality of the operation
        token_count: Number of tokens generated
        operation: Operation type ('encode' or 'decode')
        timestamp: When the operation occurred
        user_id: User identifier (for multi-tenant systems)
        metadata: Additional information
    """

    modality: Modality
    token_count: int
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "modality": self.modality.value,
            "token_count": self.token_count,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


class TokenCounter:
    """
    Token counter with modality-aware billing support.

    Usage:
        counter = TokenCounter()

        # Count tokens for an image
        image = torch.randn(1, 3, 256, 256)
        token_count = counter.count_image_tokens(
            image, spatial_compression=16, mode="DI"
        )

        # Record usage
        counter.record_usage(
            modality=Modality.IMAGE,
            token_count=token_count,
            operation="encode",
            user_id="user-123",
        )

        # Get total usage
        total = counter.get_total_tokens()
        by_modality = counter.get_usage_by_modality()
    """

    def __init__(self):
        """Initialize token counter."""
        self._usage_records: list[TokenUsage] = []
        self._total_tokens: Dict[Modality, int] = {
            Modality.IMAGE: 0,
            Modality.VIDEO: 0,
            Modality.AUDIO: 0,
            Modality.TEXT: 0,
        }

    def count_image_tokens(
        self,
        image: torch.Tensor,
        spatial_compression: int = 16,
        mode: str = "DI",
        z_channels: int = 16,
    ) -> int:
        """
        Count tokens for an image without full tokenization.

        Args:
            image: Image tensor (B, 3, H, W)
            spatial_compression: Spatial compression factor (8 or 16)
            mode: Tokenizer mode (DI or CI)
            z_channels: Number of latent channels for CI mode

        Returns:
            Expected token count
        """
        batch, _, height, width = image.shape
        h = height // spatial_compression
        w = width // spatial_compression

        if mode == "DI":
            # Discrete: (B, h, w)
            return batch * h * w
        else:
            # Continuous: (B, z_channels, h, w)
            return batch * z_channels * h * w

    def count_video_tokens(
        self,
        video: torch.Tensor,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        mode: str = "DV",
        z_channels: int = 16,
    ) -> int:
        """
        Count tokens for a video without full tokenization.

        Args:
            video: Video tensor (B, 3, T, H, W)
            spatial_compression: Spatial compression factor (8 or 16)
            temporal_compression: Temporal compression factor (4 or 8)
            mode: Tokenizer mode (DV or CV)
            z_channels: Number of latent channels for CV mode

        Returns:
            Expected token count
        """
        batch, _, frames, height, width = video.shape
        t = frames // temporal_compression
        h = height // spatial_compression
        w = width // spatial_compression

        if mode == "DV":
            # Discrete: (B, t, h, w)
            return batch * t * h * w
        else:
            # Continuous: (B, z_channels, t, h, w)
            return batch * z_channels * t * h * w

    def count_tokens(
        self,
        input_data: torch.Tensor,
        modality: Modality,
        **kwargs,
    ) -> int:
        """
        Count tokens for any modality.

        Args:
            input_data: Input tensor
            modality: Modality type
            **kwargs: Modality-specific parameters

        Returns:
            Expected token count

        Raises:
            ValueError: If modality not supported
        """
        if modality == Modality.IMAGE:
            return self.count_image_tokens(input_data, **kwargs)
        elif modality == Modality.VIDEO:
            return self.count_video_tokens(input_data, **kwargs)
        else:
            raise ValueError(
                f"Token counting not implemented for {modality.value}. "
                f"Supported: IMAGE, VIDEO"
            )

    def record_usage(
        self,
        modality: Modality,
        token_count: int,
        operation: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> TokenUsage:
        """
        Record token usage.

        Args:
            modality: Modality of the operation
            token_count: Number of tokens
            operation: Operation type ('encode' or 'decode')
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            TokenUsage record
        """
        usage = TokenUsage(
            modality=modality,
            token_count=token_count,
            operation=operation,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._usage_records.append(usage)
        self._total_tokens[modality] += token_count

        return usage

    def get_total_tokens(self) -> int:
        """Get total tokens across all modalities."""
        return sum(self._total_tokens.values())

    def get_usage_by_modality(self) -> Dict[str, int]:
        """Get token usage broken down by modality."""
        return {
            modality.value: count for modality, count in self._total_tokens.items()
        }

    def get_usage_by_user(self) -> Dict[str, int]:
        """Get token usage broken down by user."""
        user_totals: Dict[str, int] = {}

        for record in self._usage_records:
            user_id = record.user_id or "anonymous"
            user_totals[user_id] = user_totals.get(user_id, 0) + record.token_count

        return user_totals

    def get_usage_by_operation(self) -> Dict[str, int]:
        """Get token usage broken down by operation type."""
        op_totals: Dict[str, int] = {}

        for record in self._usage_records:
            op = record.operation
            op_totals[op] = op_totals.get(op, 0) + record.token_count

        return op_totals

    def get_usage_records(
        self,
        modality: Optional[Modality] = None,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> list[TokenUsage]:
        """
        Get filtered usage records.

        Args:
            modality: Filter by modality
            user_id: Filter by user
            operation: Filter by operation type

        Returns:
            List of matching TokenUsage records
        """
        records = self._usage_records

        if modality is not None:
            records = [r for r in records if r.modality == modality]

        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]

        if operation is not None:
            records = [r for r in records if r.operation == operation]

        return records

    def get_summary(self) -> Dict:
        """
        Get comprehensive usage summary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_tokens": self.get_total_tokens(),
            "by_modality": self.get_usage_by_modality(),
            "by_user": self.get_usage_by_user(),
            "by_operation": self.get_usage_by_operation(),
            "total_records": len(self._usage_records),
        }

    def reset(self):
        """Clear all usage records and counters."""
        self._usage_records.clear()
        self._total_tokens = {
            Modality.IMAGE: 0,
            Modality.VIDEO: 0,
            Modality.AUDIO: 0,
            Modality.TEXT: 0,
        }

    def export_records(self) -> list[Dict]:
        """Export all usage records as dictionaries."""
        return [record.to_dict() for record in self._usage_records]

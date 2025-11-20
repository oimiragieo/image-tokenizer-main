"""
Modality-aware tokenization architecture for enterprise-scale multimodal systems.

This module provides:
1. Base interface for all tokenizers (image, video, audio, text)
2. Registry pattern for dynamic tokenizer selection
3. Standardized tokenize/detokenize API
4. Token counting and metadata extraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch


class Modality(str, Enum):
    """Supported modalities for tokenization."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


@dataclass
class TokenizationResult:
    """
    Standardized result from tokenization operations.

    Attributes:
        tokens: Encoded tokens (discrete indices or continuous embeddings)
        modality: The modality of the input
        token_count: Number of tokens generated
        metadata: Additional information (compression ratio, dimensions, etc.)
        version: Tokenizer version used
    """

    tokens: Union[torch.Tensor, Any]
    modality: Modality
    token_count: int
    metadata: Dict[str, Any]
    version: str


@dataclass
class DetokenizationResult:
    """
    Standardized result from detokenization operations.

    Attributes:
        output: Reconstructed data (image, video, audio, text)
        modality: The modality of the output
        metadata: Additional information (reconstruction quality, metrics, etc.)
        version: Tokenizer version used
    """

    output: Union[torch.Tensor, Any]
    modality: Modality
    metadata: Dict[str, Any]
    version: str


class ModalityAwareTokenizer(ABC):
    """
    Abstract base class for all modality-aware tokenizers.

    This interface ensures consistency across image, video, audio, and text tokenizers,
    enabling unified routing, billing, and monitoring in enterprise systems.

    Design principles:
    - Consistent API: tokenize(input, **kwargs) → TokenizationResult
    - Standardized metadata: version, token_count, compression_ratio
    - Safety hooks: validate_input() called before processing
    - Performance tracking: get_performance_metrics()
    """

    def __init__(
        self,
        modality: Modality,
        version: str = "1.0.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Initialize the tokenizer.

        Args:
            modality: The modality this tokenizer handles
            version: Tokenizer version (for backward compatibility)
            device: Device to run on ('cuda', 'cpu', 'mps')
            **kwargs: Additional tokenizer-specific configuration
        """
        self.modality = modality
        self.version = version
        self.device = device
        self._stats = {
            "total_tokenizations": 0,
            "total_detokenizations": 0,
            "total_tokens_generated": 0,
        }

    @abstractmethod
    def tokenize(
        self,
        input_data: Union[torch.Tensor, Any],
        **kwargs,
    ) -> TokenizationResult:
        """
        Tokenize input data into discrete or continuous tokens.

        Args:
            input_data: Input to tokenize (image, video, audio, text)
            **kwargs: Modality-specific options (e.g., patch_size, stride)

        Returns:
            TokenizationResult with tokens, metadata, and token count

        Raises:
            ValueError: If input validation fails
        """
        pass

    @abstractmethod
    def detokenize(
        self,
        tokens: Union[torch.Tensor, Any],
        **kwargs,
    ) -> DetokenizationResult:
        """
        Reconstruct original data from tokens.

        Args:
            tokens: Tokens to decode
            **kwargs: Modality-specific options

        Returns:
            DetokenizationResult with reconstructed output and metadata

        Raises:
            ValueError: If token validation fails
        """
        pass

    @abstractmethod
    def validate_input(
        self,
        input_data: Union[torch.Tensor, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input before tokenization.

        Args:
            input_data: Input to validate

        Returns:
            (is_valid, error_message) tuple
        """
        pass

    @abstractmethod
    def get_token_count(
        self,
        input_data: Union[torch.Tensor, Any],
    ) -> int:
        """
        Calculate token count without full tokenization (for billing/routing).

        Args:
            input_data: Input to estimate

        Returns:
            Expected token count
        """
        pass

    @abstractmethod
    def get_compression_ratio(
        self,
        input_data: Union[torch.Tensor, Any],
    ) -> float:
        """
        Calculate compression ratio for input.

        Args:
            input_data: Input to analyze

        Returns:
            Compression ratio (original_size / tokenized_size)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get tokenizer metadata (version, config, capabilities).

        Returns:
            Dictionary with tokenizer metadata
        """
        return {
            "modality": self.modality.value,
            "version": self.version,
            "device": self.device,
            "class": self.__class__.__name__,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_tokenizations": self._stats["total_tokenizations"],
            "total_detokenizations": self._stats["total_detokenizations"],
            "total_tokens_generated": self._stats["total_tokens_generated"],
        }

    def _update_stats(self, operation: str, token_count: int = 0):
        """Update internal statistics."""
        if operation == "tokenize":
            self._stats["total_tokenizations"] += 1
            self._stats["total_tokens_generated"] += token_count
        elif operation == "detokenize":
            self._stats["total_detokenizations"] += 1


class TokenizerRegistry:
    """
    Registry for managing multiple tokenizers across modalities.

    Usage:
        registry = TokenizerRegistry()
        registry.register(Modality.IMAGE, ImageTokenizer(checkpoint="..."))
        registry.register(Modality.VIDEO, VideoTokenizer(checkpoint="..."))

        # Dynamic routing
        tokenizer = registry.get(Modality.IMAGE)
        result = tokenizer.tokenize(image)
    """

    def __init__(self):
        self._tokenizers: Dict[Modality, ModalityAwareTokenizer] = {}

    def register(
        self,
        modality: Modality,
        tokenizer: ModalityAwareTokenizer,
    ) -> None:
        """
        Register a tokenizer for a specific modality.

        Args:
            modality: Modality to register for
            tokenizer: Tokenizer instance

        Raises:
            ValueError: If modality already registered
        """
        if modality in self._tokenizers:
            raise ValueError(
                f"Tokenizer for {modality.value} already registered. "
                f"Use unregister() first."
            )

        if tokenizer.modality != modality:
            raise ValueError(
                f"Tokenizer modality mismatch: expected {modality.value}, "
                f"got {tokenizer.modality.value}"
            )

        self._tokenizers[modality] = tokenizer

    def unregister(self, modality: Modality) -> None:
        """Remove a tokenizer from the registry."""
        if modality in self._tokenizers:
            del self._tokenizers[modality]

    def get(self, modality: Modality) -> ModalityAwareTokenizer:
        """
        Get tokenizer for a specific modality.

        Args:
            modality: Modality to get tokenizer for

        Returns:
            Registered tokenizer

        Raises:
            KeyError: If modality not registered
        """
        if modality not in self._tokenizers:
            raise KeyError(
                f"No tokenizer registered for {modality.value}. "
                f"Available: {list(self._tokenizers.keys())}"
            )
        return self._tokenizers[modality]

    def has(self, modality: Modality) -> bool:
        """Check if modality has a registered tokenizer."""
        return modality in self._tokenizers

    def list_modalities(self) -> list[Modality]:
        """Get list of registered modalities."""
        return list(self._tokenizers.keys())

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered tokenizers."""
        return {
            modality.value: tokenizer.get_metadata()
            for modality, tokenizer in self._tokenizers.items()
        }


# Global registry instance
_global_registry = TokenizerRegistry()


def register_tokenizer(modality: Modality, tokenizer: ModalityAwareTokenizer) -> None:
    """Register a tokenizer in the global registry."""
    _global_registry.register(modality, tokenizer)


def get_tokenizer(modality: Modality) -> ModalityAwareTokenizer:
    """Get a tokenizer from the global registry."""
    return _global_registry.get(modality)


def has_tokenizer(modality: Modality) -> bool:
    """Check if modality has a registered tokenizer."""
    return _global_registry.has(modality)


def list_available_modalities() -> list[Modality]:
    """List all available modalities in the global registry."""
    return _global_registry.list_modalities()

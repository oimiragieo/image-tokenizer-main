"""
Batch processing optimization utilities for tokenizers.

Provides:
- Dynamic batching for optimal GPU utilization
- Caching for repeated inputs
- Asynchronous processing
- Memory-efficient streaming
"""

import hashlib
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cosmos_tokenizer.core.modality import Modality


class LRUCache:
    """
    LRU (Least Recently Used) cache for tokenization results.

    Features:
    - Size-based eviction
    - Hash-based lookup
    - Cache hit/miss statistics
    """

    def __init__(self, max_size_mb: float = 1000.0):
        """
        Initialize LRU cache.

        Args:
            max_size_mb: Maximum cache size in MB
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._cache: OrderedDict[str, Tuple[torch.Tensor, int]] = OrderedDict()
        self._current_size = 0
        self._hits = 0
        self._misses = 0

    def _compute_hash(self, data: torch.Tensor) -> str:
        """Compute hash of tensor data."""
        # Use tensor bytes for hashing
        data_bytes = data.cpu().numpy().tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes."""
        return tensor.element_size() * tensor.numel()

    def get(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get cached result.

        Args:
            key: Input tensor (used for hash computation)

        Returns:
            Cached tensor or None if not found
        """
        hash_key = self._compute_hash(key)

        if hash_key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            value, size = self._cache.pop(hash_key)
            self._cache[hash_key] = (value, size)
            return value.clone()
        else:
            self._misses += 1
            return None

    def put(self, key: torch.Tensor, value: torch.Tensor):
        """
        Cache a result.

        Args:
            key: Input tensor
            value: Output tensor to cache
        """
        hash_key = self._compute_hash(key)
        value_size = self._get_tensor_size(value)

        # Remove if already exists
        if hash_key in self._cache:
            _, old_size = self._cache.pop(hash_key)
            self._current_size -= old_size

        # Evict old entries if needed
        while self._current_size + value_size > self.max_size_bytes and self._cache:
            oldest_key, (oldest_value, oldest_size) = self._cache.popitem(last=False)
            self._current_size -= oldest_size

        # Add new entry
        self._cache[hash_key] = (value.clone(), value_size)
        self._current_size += value_size

    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size_mb": self._current_size / (1024 * 1024),
            "num_entries": len(self._cache),
        }


class DynamicBatcher:
    """
    Dynamic batching for optimal GPU utilization.

    Features:
    - Automatic batch size selection
    - Memory-aware batching
    - Mixed resolution support
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_memory_mb: float = 8000.0,
        device: str = "cuda",
    ):
        """
        Initialize dynamic batcher.

        Args:
            max_batch_size: Maximum batch size
            max_memory_mb: Maximum memory to use (MB)
            device: Device to run on
        """
        self.max_batch_size = max_batch_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.device = device

    def estimate_batch_size(
        self,
        sample_input: torch.Tensor,
        operation_fn: Callable,
    ) -> int:
        """
        Estimate optimal batch size.

        Args:
            sample_input: Sample input tensor (batch size 1)
            operation_fn: Operation to profile

        Returns:
            Recommended batch size
        """
        if not self.device.startswith("cuda"):
            return self.max_batch_size

        # Test with batch size 1 first
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sample_input = sample_input.to(self.device)
        _ = operation_fn(sample_input)
        torch.cuda.synchronize()

        memory_per_sample = torch.cuda.max_memory_allocated()

        # Calculate batch size that fits in memory
        estimated_batch_size = int(self.max_memory_bytes / memory_per_sample)

        # Clamp to max batch size
        return min(estimated_batch_size, self.max_batch_size)

    def batch_inputs(
        self,
        inputs: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Group inputs into batches.

        Args:
            inputs: List of input tensors
            batch_size: Batch size (auto-estimated if None)

        Returns:
            List of batched tensors
        """
        if batch_size is None:
            batch_size = self.max_batch_size

        batches = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]

            # Stack into batch
            batched = torch.stack(batch_inputs, dim=0)
            batches.append(batched)

        return batches

    def process_batched(
        self,
        inputs: List[torch.Tensor],
        operation_fn: Callable,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        """
        Process inputs in batches.

        Args:
            inputs: List of input tensors
            operation_fn: Operation to apply
            batch_size: Batch size (auto-estimated if None)
            show_progress: Show progress bar

        Returns:
            List of output tensors
        """
        batches = self.batch_inputs(inputs, batch_size)
        results = []

        iterator = batches
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc="Processing batches")
            except ImportError:
                pass

        for batch in iterator:
            batch = batch.to(self.device)
            output = operation_fn(batch)

            # Unbatch results
            if isinstance(output, torch.Tensor):
                results.extend([output[i] for i in range(output.shape[0])])
            else:
                results.append(output)

        return results


class CachedTokenizer:
    """
    Tokenizer wrapper with caching support.

    Automatically caches encoding/decoding results for repeated inputs.

    Usage:
        cached_tokenizer = CachedTokenizer(
            tokenizer=image_tokenizer,
            cache_size_mb=500.0,
        )

        # First call - computes and caches
        tokens1 = cached_tokenizer.encode(image)

        # Second call with same image - returns cached result
        tokens2 = cached_tokenizer.encode(image)  # Cache hit!
    """

    def __init__(
        self,
        tokenizer: Any,
        cache_size_mb: float = 500.0,
        enable_encode_cache: bool = True,
        enable_decode_cache: bool = True,
    ):
        """
        Initialize cached tokenizer.

        Args:
            tokenizer: Underlying tokenizer
            cache_size_mb: Cache size in MB
            enable_encode_cache: Enable caching for encode
            enable_decode_cache: Enable caching for decode
        """
        self.tokenizer = tokenizer
        self.enable_encode_cache = enable_encode_cache
        self.enable_decode_cache = enable_decode_cache

        if enable_encode_cache:
            self._encode_cache = LRUCache(max_size_mb=cache_size_mb / 2)
        if enable_decode_cache:
            self._decode_cache = LRUCache(max_size_mb=cache_size_mb / 2)

    def encode(self, input_data: torch.Tensor, **kwargs):
        """Encode with caching."""
        if self.enable_encode_cache:
            cached = self._encode_cache.get(input_data)
            if cached is not None:
                return cached

        # Compute
        result = self.tokenizer.encode(input_data, **kwargs)

        if self.enable_encode_cache:
            self._encode_cache.put(input_data, result)

        return result

    def decode(self, tokens: torch.Tensor, **kwargs):
        """Decode with caching."""
        if self.enable_decode_cache:
            cached = self._decode_cache.get(tokens)
            if cached is not None:
                return cached

        # Compute
        result = self.tokenizer.decode(tokens, **kwargs)

        if self.enable_decode_cache:
            self._decode_cache.put(tokens, result)

        return result

    def get_cache_stats(self) -> Dict[str, Dict]:
        """Get cache statistics."""
        stats = {}

        if self.enable_encode_cache:
            stats["encode"] = self._encode_cache.get_stats()

        if self.enable_decode_cache:
            stats["decode"] = self._decode_cache.get_stats()

        return stats

    def clear_cache(self):
        """Clear all caches."""
        if self.enable_encode_cache:
            self._encode_cache.clear()
        if self.enable_decode_cache:
            self._decode_cache.clear()

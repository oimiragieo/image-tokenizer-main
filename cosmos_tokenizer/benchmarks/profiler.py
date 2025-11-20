"""
Performance profiling and benchmarking for tokenizers.

Provides:
- Latency measurement
- Throughput calculation
- Memory profiling
- Batch performance analysis
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from cosmos_tokenizer.core.modality import Modality


@dataclass
class BenchmarkResult:
    """
    Result from performance benchmarking.

    Attributes:
        modality: Modality being benchmarked
        operation: Operation type ('encode' or 'decode')
        batch_size: Batch size used
        input_shape: Shape of input tensor
        token_count: Number of tokens generated
        latency_ms: Average latency in milliseconds
        throughput_tokens_per_sec: Tokens processed per second
        throughput_samples_per_sec: Samples processed per second
        memory_allocated_mb: Peak memory allocated (MB)
        memory_reserved_mb: Peak memory reserved (MB)
        device: Device used
        num_iterations: Number of iterations averaged
        warmup_iterations: Number of warmup iterations
    """

    modality: Modality
    operation: str
    batch_size: int
    input_shape: tuple
    token_count: int
    latency_ms: float
    throughput_tokens_per_sec: float
    throughput_samples_per_sec: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    device: str
    num_iterations: int = 100
    warmup_iterations: int = 10
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "modality": self.modality.value,
            "operation": self.operation,
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "device": self.device,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.modality.value.upper()} {self.operation.upper()} Benchmark\n"
            f"  Shape: {self.input_shape}, Batch: {self.batch_size}\n"
            f"  Latency: {self.latency_ms:.2f}ms\n"
            f"  Throughput: {self.throughput_samples_per_sec:.1f} samples/sec, "
            f"{self.throughput_tokens_per_sec:.1f} tokens/sec\n"
            f"  Memory: {self.memory_allocated_mb:.1f}MB allocated, "
            f"{self.memory_reserved_mb:.1f}MB reserved\n"
            f"  Device: {self.device}"
        )


class PerformanceProfiler:
    """
    Performance profiling for tokenizers.

    Features:
    - Latency and throughput measurement
    - Memory profiling
    - Multi-batch analysis
    - Comparative benchmarking

    Usage:
        profiler = PerformanceProfiler()

        # Benchmark image encoding
        result = profiler.benchmark_operation(
            operation_fn=tokenizer.encode,
            input_data=image,
            modality=Modality.IMAGE,
            operation="encode",
            num_iterations=100,
        )

        print(result)
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize profiler.

        Args:
            device: Device to profile on (cuda, cpu, mps)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

    def benchmark_operation(
        self,
        operation_fn: Callable,
        input_data: torch.Tensor,
        modality: Modality,
        operation: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        token_count: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Benchmark a tokenizer operation.

        Args:
            operation_fn: Function to benchmark (e.g., tokenizer.encode)
            input_data: Input tensor
            modality: Modality type
            operation: Operation name
            num_iterations: Number of iterations to average
            warmup_iterations: Number of warmup iterations
            token_count: Expected token count (if known)
            **kwargs: Additional arguments to operation_fn

        Returns:
            BenchmarkResult with performance metrics
        """
        # Move to device
        input_data = input_data.to(self.device)

        # Warmup
        for _ in range(warmup_iterations):
            _ = operation_fn(input_data, **kwargs)

        # Clear memory stats
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        latencies = []

        for _ in range(num_iterations):
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            output = operation_fn(input_data, **kwargs)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_latency_ms = sum(latencies) / len(latencies)
        batch_size = input_data.shape[0]

        # Calculate token count if not provided
        if token_count is None:
            if hasattr(output, "numel"):
                token_count = output.numel()
            else:
                token_count = 0

        # Calculate throughput
        throughput_samples_per_sec = (batch_size / avg_latency_ms) * 1000
        throughput_tokens_per_sec = (token_count / avg_latency_ms) * 1000

        # Memory stats
        if self.device.startswith("cuda"):
            memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            memory_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0

        return BenchmarkResult(
            modality=modality,
            operation=operation,
            batch_size=batch_size,
            input_shape=tuple(input_data.shape),
            token_count=token_count,
            latency_ms=avg_latency_ms,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            throughput_samples_per_sec=throughput_samples_per_sec,
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved,
            device=self.device,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
        )

    def benchmark_batch_sizes(
        self,
        operation_fn: Callable,
        sample_input: torch.Tensor,
        modality: Modality,
        operation: str,
        batch_sizes: List[int] = None,
        num_iterations: int = 100,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """
        Benchmark multiple batch sizes.

        Args:
            operation_fn: Function to benchmark
            sample_input: Single sample input (will be batched)
            modality: Modality type
            operation: Operation name
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            **kwargs: Additional arguments to operation_fn

        Returns:
            List of BenchmarkResults for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        results = []

        for batch_size in batch_sizes:
            # Create batched input
            if sample_input.shape[0] == 1:
                batched_input = sample_input.repeat(batch_size, *([1] * (sample_input.ndim - 1)))
            else:
                # Already batched, skip if doesn't match
                if sample_input.shape[0] != batch_size:
                    continue
                batched_input = sample_input

            try:
                result = self.benchmark_operation(
                    operation_fn=operation_fn,
                    input_data=batched_input,
                    modality=modality,
                    operation=operation,
                    num_iterations=num_iterations,
                    **kwargs,
                )
                results.append(result)

            except RuntimeError as e:
                # Handle OOM or other runtime errors
                print(f"Batch size {batch_size} failed: {e}")
                break

        return results

    def compare_operations(
        self,
        operations: Dict[str, Callable],
        input_data: torch.Tensor,
        modality: Modality,
        num_iterations: int = 100,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple operations.

        Args:
            operations: Dictionary of {operation_name: operation_fn}
            input_data: Input tensor
            modality: Modality type
            num_iterations: Number of iterations per operation

        Returns:
            Dictionary of {operation_name: BenchmarkResult}
        """
        results = {}

        for op_name, op_fn in operations.items():
            result = self.benchmark_operation(
                operation_fn=op_fn,
                input_data=input_data,
                modality=modality,
                operation=op_name,
                num_iterations=num_iterations,
            )
            results[op_name] = result

        return results

    def profile_memory(
        self,
        operation_fn: Callable,
        input_data: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Profile memory usage for an operation.

        Args:
            operation_fn: Function to profile
            input_data: Input tensor

        Returns:
            Dictionary with memory statistics (MB)
        """
        if not self.device.startswith("cuda"):
            return {
                "initial_allocated": 0.0,
                "peak_allocated": 0.0,
                "peak_reserved": 0.0,
            }

        input_data = input_data.to(self.device)

        # Clear cache and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_allocated = torch.cuda.memory_allocated() / (1024 * 1024)

        # Run operation
        _ = operation_fn(input_data)
        torch.cuda.synchronize()

        peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

        return {
            "initial_allocated_mb": initial_allocated,
            "peak_allocated_mb": peak_allocated,
            "peak_reserved_mb": peak_reserved,
            "delta_allocated_mb": peak_allocated - initial_allocated,
        }

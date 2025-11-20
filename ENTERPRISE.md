# Enterprise Features Guide

This guide covers the enterprise-grade features added to Cosmos Tokenizer for production-scale deployments.

## Table of Contents

1. [Overview](#overview)
2. [Modality-Aware Architecture](#modality-aware-architecture)
3. [Token Counting & Billing](#token-counting--billing)
4. [Usage Tracking & Analytics](#usage-tracking--analytics)
5. [Safety & Validation](#safety--validation)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Batch Processing Optimization](#batch-processing-optimization)
8. [Versioning & Metadata](#versioning--metadata)
9. [Integration Examples](#integration-examples)

---

## Overview

The enterprise features provide production-ready capabilities for large-scale multimodal tokenization systems:

- **Modality-aware architecture**: Unified interface for image, video, audio, and text tokenizers
- **Token counting & billing**: Track usage and calculate costs across modalities
- **Usage tracking**: Real-time performance metrics and analytics
- **Safety validation**: Input validation, rate limiting, and NSFW detection hooks
- **Performance benchmarking**: Latency, throughput, and quality metrics
- **Batch processing**: Caching, dynamic batching, and memory optimization
- **Versioning**: Model metadata management and backward compatibility

---

## Modality-Aware Architecture

### Unified Tokenizer Interface

All tokenizers implement a common `ModalityAwareTokenizer` base interface:

```python
from cosmos_tokenizer.core.modality import Modality
from cosmos_tokenizer.enterprise import EnterpriseImageTokenizer

# Create tokenizer
tokenizer = EnterpriseImageTokenizer(
    mode="DI",
    spatial_compression=16,
    max_resolution=4096,
    max_file_size_mb=100.0,
)

# Tokenize with standardized API
import torch
image = torch.randn(1, 3, 256, 256) * 0.5  # Range [-1, 1]

result = tokenizer.tokenize(image)

print(f"Tokens: {result.tokens.shape}")
print(f"Token count: {result.token_count}")
print(f"Compression ratio: {result.metadata['compression_ratio']:.1f}x")
print(f"Modality: {result.modality.value}")
```

### Tokenizer Registry

Register and manage multiple tokenizers:

```python
from cosmos_tokenizer.core.modality import Modality, TokenizerRegistry
from cosmos_tokenizer.enterprise import EnterpriseImageTokenizer, EnterpriseVideoTokenizer

# Create registry
registry = TokenizerRegistry()

# Register tokenizers
registry.register(
    Modality.IMAGE,
    EnterpriseImageTokenizer(mode="DI", spatial_compression=16)
)

registry.register(
    Modality.VIDEO,
    EnterpriseVideoTokenizer(mode="DV", spatial_compression=16, temporal_compression=8)
)

# Dynamic routing
def tokenize_multimodal(data, modality):
    tokenizer = registry.get(modality)
    return tokenizer.tokenize(data)

# Use
image_result = tokenize_multimodal(image, Modality.IMAGE)
video_result = tokenize_multimodal(video, Modality.VIDEO)
```

---

## Token Counting & Billing

### Basic Token Counting

```python
from cosmos_tokenizer.enterprise import TokenCounter
from cosmos_tokenizer.core.modality import Modality

counter = TokenCounter()

# Count tokens for an image
image = torch.randn(1, 3, 256, 256)
token_count = counter.count_image_tokens(
    image,
    spatial_compression=16,
    mode="DI"
)
print(f"Token count: {token_count}")  # 256 tokens for 256x256 image with 16x compression

# Record usage
counter.record_usage(
    modality=Modality.IMAGE,
    token_count=token_count,
    operation="encode",
    user_id="user-123",
)

# Get summary
summary = counter.get_summary()
print(f"Total tokens: {summary['total_tokens']:,}")
print(f"By modality: {summary['by_modality']}")
print(f"By user: {summary['by_user']}")
```

### Billing Engine

```python
from cosmos_tokenizer.enterprise import BillingEngine, PricingTier

# Initialize billing engine
engine = BillingEngine(tier=PricingTier.PRO)

# Record usage
engine.record_usage(
    modality=Modality.IMAGE,
    token_count=1_000_000,
    operation="encode",
    user_id="user-123",
)

# Check quota
can_proceed, remaining, msg = engine.check_quota("user-123")
if not can_proceed:
    print(f"Quota exceeded: {msg}")

# Generate bill
bill = engine.generate_bill(user_id="user-123")
print(f"User: {bill['user_id']}")
print(f"Total tokens: {bill['total_tokens']:,}")
print(f"Total cost: ${bill['total_cost']:.2f}")
print(f"By modality: {bill['by_modality']}")
```

### Pricing Tiers

```python
from cosmos_tokenizer.enterprise import CostCalculator, PricingTier

# Free tier (no cost)
free_calc = CostCalculator(tier=PricingTier.FREE)

# Pro tier (standard pricing)
pro_calc = CostCalculator(tier=PricingTier.PRO)

# Enterprise tier (30% discount)
enterprise_calc = CostCalculator(tier=PricingTier.ENTERPRISE)

# Calculate costs
cost = pro_calc.calculate_cost(
    modality=Modality.IMAGE,
    token_count=10_000_000,  # 10M tokens
    operation="encode",
)
print(f"Cost for 10M image tokens (PRO): ${cost:.2f}")
```

**Default Pricing (PRO tier)**:
- Image encode: $0.10 per million tokens
- Image decode: $0.05 per million tokens
- Video encode: $1.00 per million tokens
- Video decode: $0.50 per million tokens

---

## Usage Tracking & Analytics

### Real-Time Metrics

```python
from cosmos_tokenizer.enterprise import UsageTracker

tracker = UsageTracker(retention_hours=24)

# Record usage
tracker.record(
    modality=Modality.IMAGE,
    operation="encode",
    token_count=256,
    latency_ms=15.5,
    throughput_tokens_per_sec=16516.0,
    memory_mb=2048.0,
    device="cuda",
    user_id="user-123",
)

# Get metrics
metrics = tracker.get_metrics()
print(f"Total operations: {metrics['count']}")
print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Average throughput: {metrics['avg_throughput']:.1f} tokens/sec")
print(f"Average memory: {metrics['avg_memory_mb']:.1f}MB")

# Metrics by modality
by_modality = tracker.get_metrics_by_modality()
for modality, stats in by_modality.items():
    print(f"{modality}: {stats['count']} operations, {stats['avg_latency_ms']:.2f}ms avg")
```

### Time-Series Analysis

```python
# Get time-series data for visualization
time_series = tracker.get_time_series(
    interval_minutes=5,
    modality=Modality.IMAGE,
)

for bucket in time_series:
    print(f"{bucket['timestamp']}: {bucket['count']} ops, {bucket['avg_latency_ms']:.2f}ms")
```

---

## Safety & Validation

### Input Validation

```python
from cosmos_tokenizer.enterprise import SafetyValidator

validator = SafetyValidator(
    max_image_resolution=4096,
    max_video_resolution=2048,
    max_video_frames=1000,
    max_file_size_mb=100.0,
    enable_nsfw_detection=False,  # Enable for production
    rate_limit_requests_per_minute=60,
)

# Validate image
image = torch.randn(1, 3, 256, 256) * 0.5
result = validator.validate_image(image, user_id="user-123")

if result.is_valid:
    print("Image is valid")
    if result.warnings:
        print(f"Warnings: {result.warnings}")
else:
    print(f"Validation failed: {result.error_message}")

# Validate video
video = torch.randn(1, 3, 16, 256, 256) * 0.5
result = validator.validate_video(video, user_id="user-123")
```

### Rate Limiting

```python
# Check rate limit before processing
can_proceed, wait_time = validator.check_rate_limit(user_id="user-123")

if not can_proceed:
    print(f"Rate limited. Try again in {wait_time:.1f} seconds")
else:
    # Process request
    result = tokenizer.tokenize(image)
```

### Audit Logging

```python
# Get audit log
logs = validator.get_audit_log(user_id="user-123")

for log in logs:
    print(f"{log['timestamp']}: {log['action']} - {log['modality']}")
```

---

## Performance Benchmarking

### Basic Benchmarking

```python
from cosmos_tokenizer.enterprise import EnterpriseImageTokenizer
from cosmos_tokenizer.benchmarks import PerformanceProfiler

tokenizer = EnterpriseImageTokenizer(mode="DI", spatial_compression=16)
profiler = PerformanceProfiler(device="cuda")

# Benchmark encoding
image = torch.randn(1, 3, 256, 256)

result = profiler.benchmark_operation(
    operation_fn=tokenizer._tokenizer.encode,
    input_data=image,
    modality=Modality.IMAGE,
    operation="encode",
    num_iterations=100,
    warmup_iterations=10,
)

print(result)
# Output:
# IMAGE ENCODE Benchmark
#   Shape: (1, 3, 256, 256), Batch: 1
#   Latency: 15.23ms
#   Throughput: 65.6 samples/sec, 16795.0 tokens/sec
#   Memory: 2048.5MB allocated, 2560.0MB reserved
#   Device: cuda
```

### Batch Size Analysis

```python
# Test multiple batch sizes
sample = torch.randn(1, 3, 256, 256)

results = profiler.benchmark_batch_sizes(
    operation_fn=tokenizer._tokenizer.encode,
    sample_input=sample,
    modality=Modality.IMAGE,
    operation="encode",
    batch_sizes=[1, 2, 4, 8, 16, 32],
    num_iterations=50,
)

for result in results:
    print(f"Batch {result.batch_size}: {result.throughput_samples_per_sec:.1f} samples/sec")
```

### Quality Metrics

```python
from cosmos_tokenizer.benchmarks import compute_all_metrics

# Encode and decode
tokens = tokenizer._tokenizer.encode(image)
reconstructed = tokenizer._tokenizer.decode(tokens)

# Compute quality metrics
metrics = compute_all_metrics(
    original=image,
    reconstructed=reconstructed,
    use_lpips=True,  # Requires: pip install lpips
)

print(metrics)
# Output: PSNR: 32.45dB, SSIM: 0.9234, MSE: 0.000123, LPIPS: 0.0456
```

---

## Batch Processing Optimization

### Caching

```python
from cosmos_tokenizer.enterprise import CachedTokenizer

# Wrap tokenizer with cache
cached_tokenizer = CachedTokenizer(
    tokenizer=tokenizer._tokenizer,
    cache_size_mb=500.0,
    enable_encode_cache=True,
    enable_decode_cache=True,
)

# First call - computes and caches
tokens1 = cached_tokenizer.encode(image)

# Second call with same image - cache hit!
tokens2 = cached_tokenizer.encode(image)

# Check cache statistics
stats = cached_tokenizer.get_cache_stats()
print(f"Encode cache: {stats['encode']['hit_rate']:.1%} hit rate")
print(f"Cache size: {stats['encode']['cache_size_mb']:.1f}MB")
```

### Dynamic Batching

```python
from cosmos_tokenizer.enterprise import DynamicBatcher

batcher = DynamicBatcher(
    max_batch_size=32,
    max_memory_mb=8000.0,
    device="cuda",
)

# Create list of images
images = [torch.randn(1, 3, 256, 256) for _ in range(100)]

# Process in optimal batches
results = batcher.process_batched(
    inputs=images,
    operation_fn=tokenizer._tokenizer.encode,
    batch_size=None,  # Auto-estimate
    show_progress=True,
)

print(f"Processed {len(results)} images")
```

---

## Versioning & Metadata

### Model Metadata

```python
from cosmos_tokenizer.enterprise import TokenizerMetadata, VersionManager

# Create metadata
metadata = TokenizerMetadata(
    version="1.0.0",
    modality="image",
    mode="DI",
    spatial_compression=16,
    codebook_size=65536,
    model_name="Cosmos-DI-16x16",
    description="Discrete image tokenizer with 16x spatial compression",
    checkpoint_path="checkpoints/cosmos_di_16x16.pt",
    backward_compatible_with=["0.9.0", "0.9.1"],
    tags=["production", "image", "discrete"],
)

# Save metadata
from pathlib import Path
metadata.to_yaml(Path("metadata/cosmos_di_16x16.yaml"))

# Load metadata
loaded = TokenizerMetadata.from_yaml(Path("metadata/cosmos_di_16x16.yaml"))
print(f"Version: {loaded.version}")
print(f"Compression: {loaded.get_compression_info()}")
```

### Version Management

```python
# Initialize version manager
manager = VersionManager(registry_path="models/registry.yaml")

# Register version
manager.register(metadata)

# Get version
v1_meta = manager.get("1.0.0")

# List versions
versions = manager.list_versions(modality="image", mode="DI")
print(f"Available versions: {versions}")

# Get latest
latest = manager.get_latest(modality="image")
print(f"Latest version: {latest.version}")

# Check compatibility
is_compatible = manager.check_compatibility("1.0.0", "0.9.0")
```

---

## Integration Examples

### Complete Enterprise Pipeline

```python
from cosmos_tokenizer.core.modality import Modality
from cosmos_tokenizer.enterprise import (
    EnterpriseImageTokenizer,
    SafetyValidator,
    BillingEngine,
    UsageTracker,
    PricingTier,
)
import torch
import time

# Initialize components
tokenizer = EnterpriseImageTokenizer(
    mode="DI",
    spatial_compression=16,
    max_resolution=4096,
)

validator = SafetyValidator(
    max_image_resolution=4096,
    rate_limit_requests_per_minute=60,
)

billing = BillingEngine(tier=PricingTier.PRO)
tracker = UsageTracker()

def process_request(image, user_id):
    """Complete enterprise pipeline for image tokenization."""

    # 1. Rate limiting
    can_proceed, wait_time = validator.check_rate_limit(user_id)
    if not can_proceed:
        return {"error": f"Rate limited. Wait {wait_time:.1f}s"}

    # 2. Safety validation
    validation = validator.validate_image(image, user_id)
    if not validation.is_valid:
        return {"error": validation.error_message}

    # 3. Check billing quota
    can_proceed, remaining, msg = billing.check_quota(user_id)
    if not can_proceed:
        return {"error": msg}

    # 4. Tokenize
    start_time = time.perf_counter()
    result = tokenizer.tokenize(image)
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 5. Record billing
    billing.record_usage(
        modality=Modality.IMAGE,
        token_count=result.token_count,
        operation="encode",
        user_id=user_id,
    )

    # 6. Track usage
    throughput = result.token_count / (latency_ms / 1000)
    tracker.record(
        modality=Modality.IMAGE,
        operation="encode",
        token_count=result.token_count,
        latency_ms=latency_ms,
        throughput_tokens_per_sec=throughput,
        user_id=user_id,
    )

    return {
        "tokens": result.tokens,
        "token_count": result.token_count,
        "latency_ms": latency_ms,
        "metadata": result.metadata,
    }

# Use pipeline
image = torch.randn(1, 3, 256, 256) * 0.5
response = process_request(image, user_id="user-123")

if "error" in response:
    print(f"Error: {response['error']}")
else:
    print(f"Success: {response['token_count']} tokens in {response['latency_ms']:.2f}ms")
```

### LLM Gateway Integration

```python
from cosmos_tokenizer.core.modality import Modality, TokenizerRegistry

class MultimodalLLMGateway:
    """LLM gateway with multimodal tokenization support."""

    def __init__(self):
        self.registry = TokenizerRegistry()
        self.token_counter = TokenCounter()
        self.billing = BillingEngine(tier=PricingTier.PRO)

    def route_request(self, data, modality, user_id):
        """Route request to appropriate tokenizer."""

        # Get tokenizer
        tokenizer = self.registry.get(modality)

        # Count tokens for routing decision
        token_count = tokenizer.get_token_count(data)

        # Select backend based on token count
        if token_count < 1000:
            backend = "fast-model"
        else:
            backend = "large-model"

        # Tokenize
        result = tokenizer.tokenize(data)

        # Record usage
        self.billing.record_usage(
            modality=modality,
            token_count=result.token_count,
            operation="encode",
            user_id=user_id,
        )

        return {
            "backend": backend,
            "tokens": result.tokens,
            "cost": self._calculate_request_cost(user_id),
        }

    def _calculate_request_cost(self, user_id):
        bill = self.billing.generate_bill(user_id)
        return bill["total_cost"]
```

---

## Performance Best Practices

1. **Use caching for repeated inputs**
   ```python
   cached_tokenizer = CachedTokenizer(tokenizer, cache_size_mb=500)
   ```

2. **Enable batch processing for multiple inputs**
   ```python
   batcher = DynamicBatcher(max_batch_size=32)
   results = batcher.process_batched(images, tokenizer.encode)
   ```

3. **Monitor performance metrics**
   ```python
   tracker = UsageTracker()
   # Record all operations
   # Review metrics periodically
   metrics = tracker.get_metrics()
   ```

4. **Implement safety validation**
   ```python
   validator = SafetyValidator(
       max_image_resolution=4096,
       rate_limit_requests_per_minute=60,
   )
   ```

5. **Track costs and quotas**
   ```python
   billing = BillingEngine(tier=PricingTier.PRO)
   # Check quota before processing
   can_proceed, _, msg = billing.check_quota(user_id)
   ```

---

## Next Steps

- Review the [API Reference](API_REFERENCE.md) for detailed API documentation
- See [Examples](EXAMPLES.md) for more code samples
- Check [FAQ](FAQ.md) for common questions
- Read [Architecture Documentation](claude.md) for system design details

For production deployments, consider:
- Setting up monitoring dashboards for metrics
- Configuring external NSFW detection services
- Implementing persistent storage for billing records
- Setting up automated backup for model metadata
- Load testing with your expected traffic patterns

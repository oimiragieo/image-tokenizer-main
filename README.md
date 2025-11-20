# Cosmos Image Tokenizer

A world-class image and video tokenizer integrating state-of-the-art compression techniques from NVIDIA Cosmos, GloTok, AdaTok, CORE, and SCAR research.

> **📋 Documentation Status:**
> - **✅ Core Architecture:** Fully implemented and production-ready
> - **⚠️ Pre-trained Weights:** Not yet available (training required)
> - **📚 Getting Started:** See [GETTING_STARTED.md](GETTING_STARTED.md) for comprehensive tutorials

## Quick Links

- **[Getting Started Guide](GETTING_STARTED.md)** - Complete beginner tutorial
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation
- **[Examples](EXAMPLES.md)** - Practical code examples
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[FAQ](FAQ.md)** - Frequently asked questions

## Overview

This tokenizer provides unprecedented compression ratios (up to 2048×) while maintaining superior reconstruction quality through:

- **Continuous and discrete modes** for diffusion and autoregressive generation
- **Causal video processing** for frame-by-frame generation
- **Multi-stage quantization** with VQ, FSQ, LFQ, and Residual FSQ
- **Spatial-temporal compression** optimized for AI model training

### Architecture Modes

- **CI (Continuous Image)**: 8×8 or 16×16 compression for diffusion models
- **DI (Discrete Image)**: 8×8 or 16×16 with 64K token vocabulary for autoregressive models
- **CV (Continuous Video)**: 4×8×8 or 8×8×8 temporal-spatial compression
- **DV (Discrete Video)**: 8×16×16 for autoregressive video generation

## Implementation Status

### ✅ Currently Implemented

- **Core Tokenizers:** CI, DI, CV, DV modes fully functional
- **Quantization Methods:** VQ, FSQ, LFQ, Residual FSQ
- **High-Level APIs:** `ImageTokenizer` and `VideoTokenizer` classes
- **Training Infrastructure:** Loss functions, optimizer support
- **Video Features:** Causal processing, sliding window for long videos
- **Performance Optimizations:** Mixed precision, JIT compilation support
- **Testing:** Unit tests for core functionality

### ⚠️ Planned/Not Yet Implemented

- **Pre-trained Checkpoints:** Coming soon (you must train your own for now)
- **CLI Tools:** Command-line interfaces (referenced in setup.py but not implemented)
- **Training Scripts:** Full end-to-end training pipelines (basic examples available)
- **Advanced Features:** Dual-codebook (GloTok), object-aware compression (AdaTok/CORE), semantic alignment (SCAR)
- **Benchmarking Suite:** Comprehensive evaluation scripts
- **Model Zoo:** Pre-trained model repository

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 11.8 (recommended for GPU support)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cosmos-image-tokenizer.git
cd cosmos-image-tokenizer

# Install dependencies
pip install -e .
```

### Installation with Optional Features

```bash
# Video support (av, opencv, etc.)
pip install -e ".[video]"

# Training support (PyTorch Lightning, W&B, etc.)
pip install -e ".[train]"

# Development tools (pytest, black, etc.)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### GPU Support

Install PyTorch with CUDA first:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install tokenizer
pip install -e .
```

## Quick Start

### Image Tokenization

```python
import torch
from cosmos_tokenizer import ImageTokenizer

# Initialize discrete image tokenizer
tokenizer = ImageTokenizer(
    mode="DI",  # Discrete Image mode
    spatial_compression=16,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prepare image (must be in range [-1, 1])
image = torch.randn(1, 3, 256, 256) * 2 - 1  # (B, C, H, W)

# Encode to discrete tokens
tokens = tokenizer.encode(image)  # (1, 16, 16) integer indices

# Decode back to image
reconstructed = tokenizer.decode(tokens)  # (1, 3, 256, 256)

# Or full round-trip
reconstructed = tokenizer(image)
```

### Video Tokenization

```python
from cosmos_tokenizer import VideoTokenizer

# Initialize continuous video tokenizer
tokenizer = VideoTokenizer(
    mode="CV",  # Continuous Video mode
    spatial_compression=8,
    temporal_compression=8,
    device="cuda"
)

# Prepare video (B, C, T, H, W) in range [-1, 1]
video = torch.randn(1, 3, 32, 256, 256) * 2 - 1

# Process with sliding window (memory-efficient)
reconstructed = tokenizer(video, temporal_window=17)
```

### Working with Real Images

```python
from PIL import Image
import numpy as np

# Load and preprocess
img = Image.open("photo.jpg").convert("RGB").resize((256, 256))
img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]
img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

# Tokenize
tokens = tokenizer.encode(img_tensor.to(tokenizer.device))

# Reconstruct
reconstructed = tokenizer.decode(tokens)

# Convert back to PIL
reconstructed = (reconstructed.squeeze(0).cpu() + 1.0) / 2.0
reconstructed = torch.clamp(reconstructed, 0, 1)
reconstructed = (reconstructed.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
result_img = Image.fromarray(reconstructed)
result_img.save("reconstructed.jpg")
```

**For more examples, see [EXAMPLES.md](EXAMPLES.md)**

## Architecture Overview

### Encoder Pipeline
```
Input Image/Video
    ↓
Haar Wavelet Patching (4×4 lossless compression)
    ↓
Progressive Downsampling
  - ResNet Blocks
  - Self-Attention @ 32px
  - Channel Multipliers: [1, 2, 4, 4]
    ↓
Quantization (VQ/FSQ/LFQ/Residual FSQ)
    ↓
Latent Representation
```

### Decoder Pipeline
```
Latent Representation
    ↓
Post-Quantization Processing
    ↓
Progressive Upsampling
  - Transposed Convolutions
  - ResNet Blocks
  - Self-Attention @ 32px
    ↓
Inverse Haar Wavelet
    ↓
Reconstructed Image/Video
```

## Training

**Note:** Pre-trained weights are not included. You must train your own models.

### Basic Training Example

```python
from cosmos_tokenizer.networks.autoencoder import DiscreteImageTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_DI_16x16
from cosmos_tokenizer.training.losses import TokenizerLoss

# Create model
model = DiscreteImageTokenizer(COSMOS_DI_16x16).cuda()

# Create loss
criterion = TokenizerLoss(
    recon_loss_type="l1",
    perceptual_weight=1.0,
    quantizer_weight=1.0,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for images in dataloader:
    images = images.cuda()

    # Forward
    reconstructed, info = model(images)

    # Compute loss
    losses = criterion(reconstructed, images, info)
    total_loss = losses["total_loss"]

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Training Configurations

See `configs/` directory for pre-configured training setups:
- `ci_8x8.yaml` - Continuous Image 8×8
- `di_16x16.yaml` - Discrete Image 16×16

**For full training guides, see [GETTING_STARTED.md](GETTING_STARTED.md) and [EXAMPLES.md](EXAMPLES.md)**

## Compression Performance

| Mode | Input Size | Token Size | Compression Ratio |
|------|------------|------------|-------------------|
| DI 8×8 | 256×256×3 | 32×32 | 64× spatial |
| DI 16×16 | 256×256×3 | 16×16 | 256× spatial |
| DV 8×16×16 | 256×256×32×3 | 16×16×4 | 2048× total |

**Note:** Actual quality depends on training. Without pre-trained weights, reconstructions will be poor until you train the model.

## Docker Support

```bash
# Build container
docker build -t cosmos-tokenizer .

# Run interactive session
docker run --gpus all -it cosmos-tokenizer

# Mount data directory
docker run --gpus all -v $(pwd)/data:/workspace/data cosmos-tokenizer
```

## Research Integration

This implementation integrates techniques from:

1. **NVIDIA Cosmos-Tokenizer** - Production-grade architecture, causal video processing ✅ **Implemented**
2. **GloTok** - Dual-codebook architecture ⚠️ **Planned**
3. **AdaTok** - Object-aware adaptive compression ⚠️ **Planned**
4. **CORE** - Object-centric token merging ⚠️ **Planned**
5. **SCAR** - Semantic alignment guidance ⚠️ **Planned**

**Currently implemented:** Core NVIDIA Cosmos architecture with multiple quantization methods (VQ, FSQ, LFQ, Residual FSQ).

**Planned:** Advanced features from other papers (dual-codebook, object-aware compression, semantic alignment).

## Citation

```bibtex
@software{cosmos_tokenizer_2025,
  title={Cosmos Image Tokenizer: World-Class Visual Compression},
  author={Cosmos Tokenizer Team},
  year={2025},
  url={https://github.com/yourusername/cosmos-image-tokenizer}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Built on research from NVIDIA and inspired by cutting-edge papers in visual tokenization. Special thanks to the open-source community for their foundational work.

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/cosmos-image-tokenizer/issues)
- **Documentation:** See the [documentation links](#quick-links) above
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/cosmos-image-tokenizer/discussions)

---

**⭐ Star this repo** if you find it useful!

**📖 Read [GETTING_STARTED.md](GETTING_STARTED.md)** for a comprehensive tutorial.

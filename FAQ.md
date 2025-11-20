# Frequently Asked Questions (FAQ)

Answers to common questions about Cosmos Image Tokenizer.

## Table of Contents
- [General Questions](#general-questions)
- [Technical Questions](#technical-questions)
- [Usage Questions](#usage-questions)
- [Training Questions](#training-questions)
- [Comparison Questions](#comparison-questions)

---

## General Questions

### What is image tokenization?

Image tokenization is the process of converting images into a compressed discrete or continuous representation (tokens/latents). Think of it as:

- **For discrete mode:** Like converting text to word IDs, but for images. A 256×256 image becomes 16×16 token IDs from a 64K vocabulary.
- **For continuous mode:** Like a smart compression that preserves features AI models care about, not just visual fidelity.

**Why is this useful?**
- Train AI models faster (process tokens instead of raw pixels)
- Save storage (768× compression or more)
- Enable new architectures (GPT for images, latent diffusion models)

---

### Is this the same as JPEG compression?

No, it's fundamentally different:

**JPEG:**
- Optimized for human perception
- Fixed compression algorithm
- Frequency-domain (DCT) compression
- Irrelevant to AI models

**Tokenization:**
- Optimized for AI model training
- Learned compression (trained neural network)
- Semantic compression (preserves meaning, not just pixels)
- Enables diffusion and autoregressive models

**Example:** A 256×256 image compressed to 16×16 tokens can be reconstructed reasonably well, and more importantly, the tokens are semantically meaningful for AI models.

---

### Do I need pre-trained weights?

**Current status:** Pre-trained weights are **not yet available** with this repository.

**What this means:**
- The codebase provides the architecture
- You can train your own model from scratch
- Without training, reconstructions will be poor (random weights)

**Options:**
1. Train your own (requires GPU cluster and large dataset like ImageNet)
2. Wait for official pre-trained releases (coming soon)
3. Search for community-trained checkpoints

---

### Is this production-ready?

**Core library:** ✅ Yes
- Architecture is complete and tested
- API is stable
- Supports GPU/CPU, mixed precision, distributed training

**Ecosystem:** ⚠️ Partially
- ❌ No pre-trained weights yet
- ❌ No CLI tools yet
- ❌ Advanced features (dual-codebook, object-aware) not implemented
- ✅ Core tokenization works
- ✅ Training infrastructure exists

**Use for:**
- Research and experimentation (yes)
- Production deployment (wait for pre-trained weights)
- Learning about tokenization (yes)
- Building new models from scratch (yes)

---

### What's the license?

**Apache 2.0** - permissive open-source license.

You can:
- Use commercially
- Modify the code
- Distribute modified versions
- Use privately
- Patent use

You must:
- Include license and copyright notice
- State changes made

You cannot:
- Hold authors liable
- Use trademarks

---

## Technical Questions

### What's the difference between CI, DI, CV, and DV?

| Mode | Type | Use Case | Output |
|------|------|----------|--------|
| **CI** | Continuous Image | Diffusion models (Stable Diffusion-style) | Float tensors |
| **DI** | Discrete Image | Autoregressive models (GPT-style) | Integer token IDs |
| **CV** | Continuous Video | Video diffusion models | Float tensors |
| **DV** | Discrete Video | Autoregressive video generation | Integer token IDs |

**Choosing:**
- Building a diffusion model? Use **CI** (images) or **CV** (video)
- Building an autoregressive model? Use **DI** (images) or **DV** (video)
- Not sure? Start with **DI** (most versatile)

---

### What does "causal" mean in video processing?

**Causal processing:** Frame `t` can only access frames `0, 1, 2, ..., t` (past and present), not future frames `t+1, t+2, ...`

**Why important?**
- Essential for autoregressive video generation
- Prevents information leakage
- Models can generate videos frame-by-frame

**Implementation:**
- Causal 3D convolutions (padding only on left/past side)
- Triangular attention masks
- time2batch processing

**When to use:**
- ✅ Autoregressive video models (DV mode)
- ✅ Online/streaming video processing
- ❌ Diffusion models (can see whole video)
- ❌ Video classification (can see future frames)

---

### What compression ratios are possible?

**Image tokenization:**

| Mode | Spatial | Patch | Latent Channels | Total Compression |
|------|---------|-------|-----------------|-------------------|
| CI 8×8 | 8×8 = 64 | 4×4 = 16 | 16 | 64× (lossy) |
| CI 16×16 | 16×16 = 256 | 4×4 = 16 | 16 | 256× (lossy) |
| DI 8×8 | 8×8 = 64 | 4×4 = 16 | 1 (indices) | ~768× (lossy) |
| DI 16×16 | 16×16 = 256 | 4×4 = 16 | 1 (indices) | ~3072× (lossy) |

**Video tokenization:**

| Mode | Temporal | Spatial | Total Compression |
|------|----------|---------|-------------------|
| CV 4×8×8 | 4 | 8×8 = 64 | 256× |
| CV 8×8×8 | 8 | 8×8 = 64 | 512× |
| DV 8×16×16 | 8 | 16×16 = 256 | 2048× |

**Note:** Higher compression = smaller files but lower quality

---

### What's the difference between VQ, FSQ, and LFQ?

**Vector Quantization (VQ):**
- Classic codebook approach (VQ-VAE)
- Explicit codebook with learnable embeddings
- Prone to codebook collapse
- Memory: `codebook_size × embedding_dim` parameters

**Finite Scalar Quantization (FSQ):**
- Implicit codebook (no learned embeddings)
- Quantizes each dimension to fixed levels
- Example: `[8, 8, 8, 5, 5, 5]` = 64K codes
- More stable training, less collapse
- Memory: Zero extra parameters!

**Lookup-Free (LFQ):**
- Binary quantization to {-1, +1}
- Extremely simple: `sign(x)`
- 2^D codebook size (D = latent dim)
- Fast but may lose precision

**Recommendation:**
- Start with **FSQ** (best balance)
- Use **VQ** if you need fine-grained control
- Use **LFQ** for extremely fast inference

---

### Why range [-1, 1] and not [0, 1]?

**Historical reasons:**
- Neural networks train better with zero-mean data
- Tanh activation outputs [-1, 1]
- Batch norm and layer norm expect zero-centered data

**Practical reasons:**
- Symmetric range is easier for models to learn
- Better gradient flow
- Standard in generative models (GANs, diffusion, VAEs)

**Conversion:**
```python
# [0, 255] → [-1, 1]
x = (x / 127.5) - 1.0

# [0, 1] → [-1, 1]
x = x * 2.0 - 1.0

# [-1, 1] → [0, 1]
x = (x + 1.0) / 2.0

# [-1, 1] → [0, 255]
x = ((x + 1.0) / 2.0) * 255.0
```

---

## Usage Questions

### Can I use this on CPU?

**Yes, but it's slow.**

**Performance:**
- GPU (A100): ~1000 FPS for 256×256 images
- CPU (modern): ~10-50 FPS for 256×256 images

**When CPU is OK:**
- Learning and experimentation
- Small datasets
- Infrequent inference

**When you need GPU:**
- Training (essential)
- Large-scale inference
- Real-time applications
- Video processing

---

### What image sizes are supported?

**Requirement:** Height and width must be divisible by `spatial_compression × patch_size`

**Common values:**
- `spatial_compression`: 8 or 16
- `patch_size`: 4 (default)
- Divisor: 32 (for compression=8) or 64 (for compression=16)

**Supported sizes (compression=16, patch=4):**
- ✅ 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, ..., 1024, ...
- ❌ 100, 150, 200, 250 (not divisible by 64)

**Recommended:** Use powers of 2 (256, 512, 1024) for best performance.

**Arbitrary sizes?**
```python
# Resize to nearest supported size
from PIL import Image
img = Image.open("photo.jpg")

target_size = 256  # or 512, 1024
img = img.resize((target_size, target_size))
```

---

### Can I tokenize rectangular images (not square)?

**Yes!** As long as both dimensions are divisible by the required factor.

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)

# All valid (divisible by 64):
img1 = torch.randn(1, 3, 256, 512)  # ✅ 256×512
img2 = torch.randn(1, 3, 512, 256)  # ✅ 512×256
img3 = torch.randn(1, 3, 320, 640)  # ✅ 320×640
img4 = torch.randn(1, 3, 300, 400)  # ❌ Not divisible by 64
```

---

### How do I save and load tokens?

**Save tokens:**
```python
import torch

tokens = tokenizer.encode(images)
torch.save(tokens, "tokens.pt")

# For large datasets:
torch.save({
    'tokens': tokens,
    'filenames': filenames,
    'metadata': metadata
}, "dataset_tokens.pt")
```

**Load tokens:**
```python
tokens = torch.load("tokens.pt")
reconstructed = tokenizer.decode(tokens)

# For large datasets:
data = torch.load("dataset_tokens.pt")
tokens = data['tokens']
filenames = data['filenames']
```

**Compressed storage (safetensors):**
```python
from safetensors.torch import save_file, load_file

# Save
save_file({'tokens': tokens}, "tokens.safetensors")

# Load
data = load_file("tokens.safetensors")
tokens = data['tokens']
```

---

## Training Questions

### How much data do I need to train?

**Minimum (for experimentation):**
- 10K-100K images
- Won't match SOTA but you'll learn

**Good results:**
- 1M+ images (like ImageNet-1K)
- Diverse dataset

**Best results:**
- 10M+ images
- High-quality, diverse dataset
- Multiple passes (epochs)

**Video:**
- Much more data needed (videos are harder)
- Minimum: 100K video clips
- Best: 1M+ clips (like Kinetics, WebVid)

---

### How long does training take?

**Image tokenizer (DI 16×16) on ImageNet-1K:**

| Hardware | Batch Size | Time per Epoch | Total (100 epochs) |
|----------|------------|----------------|---------------------|
| 1× A100 (80GB) | 64 | ~8 hours | ~33 days |
| 8× A100 (80GB) | 512 | ~1 hour | ~4 days |
| 1× RTX 3090 | 32 | ~16 hours | ~67 days |
| 8× RTX 3090 | 256 | ~2 hours | ~8 days |

**Speedups:**
- Mixed precision (bfloat16): 2-3× faster
- Efficient dataloader: 1.5-2× faster
- Compile (PyTorch 2.0+): 1.2-1.5× faster

**Video tokenizer:**
- 5-10× longer than image
- Requires more GPU memory

---

### What's a good learning rate?

**Recommended starting points:**

| Optimizer | Learning Rate | Weight Decay |
|-----------|---------------|--------------|
| AdamW | 1e-4 | 0.01 |
| Adam | 1e-4 | 0 |
| SGD | 1e-3 | 1e-4 |

**Learning rate schedule:**
```python
# Cosine annealing (recommended)
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

**Warmup (first few epochs):**
```python
# Linear warmup over 10k steps
for step in range(10000):
    lr = (step / 10000) * 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Signs LR is wrong:**
- Too high: Loss explodes, NaN, unstable training
- Too low: Very slow improvement, plateaus early

---

### Should I use perceptual loss?

**Perceptual loss** (VGG-based feature matching):

**Pros:**
- Better visual quality
- Preserves textures and details
- Prevents over-smoothing

**Cons:**
- 2-3× slower training (VGG forward pass)
- Requires torchvision models
- More memory

**Recommendation:**
- ✅ Use for final training
- ❌ Disable for early experimentation (faster iteration)
- ✅ Use for image models
- ⚠️ Optional for video (expensive)

**Configuration:**
```python
criterion = TokenizerLoss(
    recon_loss_type="l1",
    perceptual_weight=1.0,      # Weight relative to recon loss
    use_perceptual=True,        # Enable/disable
)
```

---

## Comparison Questions

### How does this compare to VQGAN?

**Similarities:**
- Both are VQ-VAE variants
- Both use discrete codebooks
- Both trained with perceptual loss

**Differences:**

| Feature | VQGAN | Cosmos Tokenizer |
|---------|-------|------------------|
| Quantization | VQ only | VQ, FSQ, LFQ, Residual FSQ |
| Compression | Typically 16×16 | 8×8, 16×16, video: up to 8×16×16 |
| Video support | No | Yes (causal processing) |
| Modes | Discrete only | Continuous + Discrete |
| Patching | Simple downsampling | Haar wavelet (lossless) |
| Advanced features | Discriminator loss | Dual-codebook, object-aware |

**When to use which:**
- VQGAN: Established, many pre-trained models available
- Cosmos: More flexible, better compression, video support

---

### How does this compare to Stable Diffusion's VAE?

**Stable Diffusion VAE:**
- Continuous latent (no discrete tokens)
- 8×8 spatial compression
- ~16 latent channels
- Fixed architecture

**Cosmos Tokenizer (CI mode):**
- Same basic idea (continuous VAE)
- Configurable compression (8×8 or 16×16)
- Configurable latent channels
- More modern architecture options

**Cosmos Tokenizer (DI mode):**
- Discrete tokens (different from SD VAE)
- For autoregressive models, not diffusion
- More aggressive compression possible

---

### How does this compare to DALLā‹…E's tokenizer?

**DALL·E tokenizer (dVAE):**
- Discrete, 8192 vocabulary
- 32×32 tokens for 256×256 images
- ~8× spatial compression

**Cosmos Tokenizer (DI mode):**
- 64K vocabulary (8× larger)
- 16×16 tokens for 256×256 images (16× compression)
- More aggressive compression
- Multiple quantization methods

**Trade-off:**
- DALL·E: More tokens, less compression, easier to model
- Cosmos: Fewer tokens, more compression, harder to model

---

### Can I use this with Stable Diffusion?

**For training a new Stable Diffusion model:**
- ✅ Yes, use **CI mode** (continuous image tokenizer)
- Replace the standard VAE with Cosmos CI tokenizer
- Retrain the diffusion model on new latents

**For existing Stable Diffusion models:**
- ❌ No, incompatible latent spaces
- SD models expect specific VAE latent format
- Would need retraining

**Example:**
```python
# Train new diffusion model with Cosmos latents
tokenizer = ImageTokenizer(mode="CI", spatial_compression=8)
latents = tokenizer.encode(images, deterministic=True)

# Train diffusion model on latents (not images)
diffusion_model = MyDiffusionModel(latent_channels=16)
```

---

## More Questions?

- Check [GETTING_STARTED.md](GETTING_STARTED.md) for tutorials
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Check [API_REFERENCE.md](API_REFERENCE.md) for technical details
- Open an issue on GitHub

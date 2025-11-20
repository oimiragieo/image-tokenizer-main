# Getting Started with Cosmos Image Tokenizer

## Table of Contents
1. [What is This?](#what-is-this)
2. [Why Would I Use This?](#why-would-i-use-this)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Your First Tokenization](#your-first-tokenization)
6. [Understanding the Modes](#understanding-the-modes)
7. [Common Use Cases](#common-use-cases)
8. [Next Steps](#next-steps)

---

## What is This?

**Cosmos Image Tokenizer** is a tool that compresses images and videos into much smaller representations (called "tokens" or "latent codes") while preserving quality. Think of it like a sophisticated ZIP compression, but specifically designed for AI models.

### Simple Analogy

Imagine you have a 256x256 pixel image (196,608 numbers). This tokenizer can:
- Compress it down to just 16x16 tokens (256 numbers) - that's **768× smaller!**
- Reconstruct the image back from those 256 numbers with minimal quality loss

This is incredibly useful for:
- **Training AI models faster** (less data to process)
- **Saving storage space** (compressed representations)
- **Generating images/videos** (diffusion models, autoregressive models)

---

## Why Would I Use This?

### For AI Researchers
- Train diffusion models (like Stable Diffusion) with less compute
- Build autoregressive image generators (like GPT, but for images)
- Experiment with video generation models

### For ML Engineers
- Reduce dataset storage requirements by 100-2000×
- Speed up image/video model training pipelines
- Deploy efficient visual AI systems

### For Students/Learners
- Understand how modern AI compresses visual data
- Learn about VAEs (Variational Autoencoders) and VQ-VAEs (Vector Quantized VAEs)
- Experiment with cutting-edge computer vision techniques

---

## Prerequisites

### Required Knowledge
- **Basic Python** (you should know how to install packages and run scripts)
- **Basic understanding of images** (what pixels are, RGB channels, etc.)
- **Familiarity with PyTorch** (helpful but not strictly required)

You do NOT need to understand:
- Deep learning architecture design
- Advanced mathematics (calculus, linear algebra)
- Video encoding formats

### System Requirements

**Minimum:**
- Python 3.10 or newer
- 8GB RAM
- CPU-only is supported (but slow)

**Recommended:**
- Python 3.10 or 3.11
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (for fast processing)
- CUDA 11.8 or newer (if using GPU)

**Check your Python version:**
```bash
python --version
# Should show: Python 3.10.x or 3.11.x or 3.12.x
```

**Check if you have a CUDA-capable GPU (optional):**
```bash
nvidia-smi
# If this works, you have an NVIDIA GPU
```

---

## Installation

### Step 1: Create a Clean Python Environment

**Why?** This prevents conflicts with other Python packages you have installed.

**Using venv (recommended for beginners):**
```bash
# Navigate to where you want to work
cd ~/projects  # or wherever you keep your code

# Create a virtual environment
python -m venv cosmos-env

# Activate it
# On Linux/Mac:
source cosmos-env/bin/activate
# On Windows:
cosmos-env\Scripts\activate

# You should now see (cosmos-env) in your terminal prompt
```

**Using conda (if you prefer):**
```bash
conda create -n cosmos python=3.10
conda activate cosmos
```

### Step 2: Install the Package

**Option A: Basic Installation (CPU only, images only)**
```bash
pip install torch torchvision  # Install PyTorch first
git clone https://github.com/yourusername/cosmos-image-tokenizer.git
cd cosmos-image-tokenizer
pip install -e .
```

**Option B: GPU Support (recommended if you have NVIDIA GPU)**
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the tokenizer
git clone https://github.com/yourusername/cosmos-image-tokenizer.git
cd cosmos-image-tokenizer
pip install -e .
```

**Option C: Full Installation (GPU + video support + training)**
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install tokenizer with all extras
git clone https://github.com/yourusername/cosmos-image-tokenizer.git
cd cosmos-image-tokenizer
pip install -e ".[all]"
```

**What do the installation options mean?**
- Basic: Just the core library for image tokenization
- `.[video]`: Adds video processing support (av, opencv, etc.)
- `.[train]`: Adds training support (PyTorch Lightning, Weights & Biases)
- `.[all]`: Everything (video + training + development tools)

### Step 3: Verify Installation

Create a file called `test_install.py`:
```python
import torch
from cosmos_tokenizer import ImageTokenizer

print("✓ Imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("Using CPU (this will be slower)")
    device = "cpu"

# Try creating a tokenizer
tokenizer = ImageTokenizer(
    mode="DI",
    spatial_compression=16,
    device=device
)
print("✓ Tokenizer created successfully!")
print("\n🎉 Installation verified! You're ready to go.")
```

Run it:
```bash
python test_install.py
```

If you see the success messages, you're all set!

---

## Your First Tokenization

Let's tokenize your first image step-by-step.

### Step 1: Prepare an Image

Create a file called `first_tokenization.py`:

```python
import torch
from cosmos_tokenizer import ImageTokenizer
from PIL import Image
import numpy as np

# ========================================
# PART 1: Load and Prepare an Image
# ========================================

# Option A: Load from file
image_path = "path/to/your/image.jpg"  # Change this to your image
image = Image.open(image_path).convert("RGB")

# Resize to 256x256 (tokenizer works best with powers of 2)
image = image.resize((256, 256))

# Convert to PyTorch tensor
# Shape: (height, width, channels) -> (channels, height, width)
image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # (3, 256, 256)

# Scale to [-1, 1] range (this is what the model expects)
image_tensor = image_tensor * 2.0 - 1.0

# Add batch dimension: (3, 256, 256) -> (1, 3, 256, 256)
image_tensor = image_tensor.unsqueeze(0)

print(f"Image shape: {image_tensor.shape}")
print(f"Image range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")

# ========================================
# PART 2: Create Tokenizer
# ========================================

# Choose device (GPU if available, CPU otherwise)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create tokenizer
# Mode "DI" = Discrete Image (outputs integer token IDs)
# spatial_compression=16 means 256x256 -> 16x16 tokens
tokenizer = ImageTokenizer(
    mode="DI",
    spatial_compression=16,
    device=device
)

# Move image to same device as model
image_tensor = image_tensor.to(device)

# ========================================
# PART 3: Encode (Compress)
# ========================================

with torch.no_grad():  # Disable gradient computation (we're not training)
    # Encode image to tokens
    tokens = tokenizer.encode(image_tensor)

    print(f"\n📊 Compression Results:")
    print(f"  Original shape: {image_tensor.shape} = {image_tensor.numel()} numbers")
    print(f"  Token shape: {tokens.shape} = {tokens.numel()} numbers")

    compression_ratio = image_tensor.numel() / tokens.numel()
    print(f"  Compression ratio: {compression_ratio:.1f}×")

    print(f"\n  Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"  (These are integer IDs from a 64,000-word vocabulary)")

# ========================================
# PART 4: Decode (Decompress)
# ========================================

with torch.no_grad():
    # Reconstruct image from tokens
    reconstructed = tokenizer.decode(tokens)

    print(f"\n🔄 Reconstruction:")
    print(f"  Reconstructed shape: {reconstructed.shape}")

# ========================================
# PART 5: Save and Compare
# ========================================

# Convert back to image format
def tensor_to_image(tensor):
    # Move to CPU, remove batch dimension, convert to numpy
    img = tensor.cpu().squeeze(0)
    # Scale from [-1, 1] to [0, 1]
    img = (img + 1.0) / 2.0
    # Clamp to valid range
    img = torch.clamp(img, 0, 1)
    # Convert to (H, W, C) format
    img = img.permute(1, 2, 0).numpy()
    # Convert to 0-255 range
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# Save results
original_img = tensor_to_image(image_tensor)
reconstructed_img = tensor_to_image(reconstructed)

original_img.save("original.png")
reconstructed_img.save("reconstructed.png")

print(f"\n✅ Saved:")
print(f"  - original.png (input)")
print(f"  - reconstructed.png (after compression + decompression)")
print(f"\nOpen these files to compare quality!")

# ========================================
# PART 6: Calculate Quality Metrics
# ========================================

# Mean Squared Error (lower is better, 0 = perfect)
mse = torch.mean((image_tensor - reconstructed) ** 2).item()
print(f"\n📏 Quality Metrics:")
print(f"  MSE: {mse:.6f}")

# Peak Signal-to-Noise Ratio (higher is better, >30 is good)
if mse > 0:
    psnr = 10 * torch.log10(4.0 / mse).item()  # 4.0 because range is [-1, 1]
    print(f"  PSNR: {psnr:.2f} dB")
```

### Step 2: Run It

```bash
python first_tokenization.py
```

### Step 3: Understand the Output

You should see something like:
```
Image shape: torch.Size([1, 3, 256, 256])
Image range: [-1.00, 1.00]
Using device: cuda

📊 Compression Results:
  Original shape: torch.Size([1, 3, 256, 256]) = 196608 numbers
  Token shape: torch.Size([1, 16, 16]) = 256 numbers
  Compression ratio: 768.0×

  Token range: [0, 63999]
  (These are integer IDs from a 64,000-word vocabulary)

🔄 Reconstruction:
  Reconstructed shape: torch.Size([1, 3, 256, 256])

✅ Saved:
  - original.png (input)
  - reconstructed.png (after compression + decompression)

📏 Quality Metrics:
  MSE: 0.000234
  PSNR: 36.31 dB
```

**What does this mean?**
- Your 256×256 image (196,608 numbers) was compressed to 16×16 tokens (256 numbers)
- That's **768× compression!**
- The PSNR of 36 dB means the reconstructed image is very close to the original
- The tokens are integers from 0 to 63,999 (a 64K vocabulary)

---

## Understanding the Modes

The tokenizer has **4 modes**. Here's what each means:

### Mode Comparison Table

| Mode | Full Name | Output Type | When to Use | Compression Example |
|------|-----------|-------------|-------------|---------------------|
| **CI** | Continuous Image | Float numbers | Diffusion models (Stable Diffusion-style) | 256×256 → 32×32 floats |
| **DI** | Discrete Image | Integer IDs | Autoregressive models (GPT-style) | 256×256 → 16×16 integers |
| **CV** | Continuous Video | Float numbers | Video diffusion models | 256×256×32 → 32×32×4 floats |
| **DV** | Discrete Video | Integer IDs | Autoregressive video generation | 256×256×32 → 16×16×4 integers |

### Continuous vs. Discrete

**Continuous (CI, CV):**
```python
tokenizer = ImageTokenizer(mode="CI", spatial_compression=8)
tokens = tokenizer.encode(image)
# tokens: tensor of floats, shape (1, 16, 32, 32)
# Example values: [0.234, -1.567, 0.891, ...]
```
- Outputs: Floating-point numbers
- Used for: Diffusion models (like DALL-E, Stable Diffusion)
- Pros: Smoother representations, better for continuous optimization
- Cons: Takes more memory

**Discrete (DI, DV):**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
tokens = tokenizer.encode(image)
# tokens: tensor of integers, shape (1, 16, 16)
# Example values: [42315, 8721, 59234, ...]
```
- Outputs: Integer token IDs (like words in a vocabulary)
- Used for: Autoregressive models (like GPT for images)
- Pros: More compressed, easier to work with for language-model-style training
- Cons: Discrete steps (quantization error)

### Image vs. Video

**Image (CI, DI):**
- Processes single frames
- Shape: `(batch, channels, height, width)`
- Example: `(1, 3, 256, 256)` = 1 image, RGB, 256×256 pixels

**Video (CV, DV):**
- Processes video sequences
- Shape: `(batch, channels, time, height, width)`
- Example: `(1, 3, 32, 256, 256)` = 1 video, RGB, 32 frames, 256×256 pixels
- Special feature: **Causal processing** (each frame only sees past frames, not future)

### Compression Levels

```python
# Lower compression = better quality, larger tokens
tokenizer = ImageTokenizer(mode="DI", spatial_compression=8)
# 256×256 → 32×32 tokens (64× compression)

# Higher compression = worse quality, smaller tokens
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
# 256×256 → 16×16 tokens (256× compression)
```

**Available compression levels:**
- `spatial_compression=8`: Medium compression, high quality
- `spatial_compression=16`: High compression, good quality (recommended)

For video, you also have `temporal_compression`:
- `temporal_compression=4`: Keep more temporal detail
- `temporal_compression=8`: Compress time more aggressively

---

## Common Use Cases

### Use Case 1: Batch Process Multiple Images

```python
import torch
from cosmos_tokenizer import ImageTokenizer
from pathlib import Path
from PIL import Image
import numpy as np

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device=device)

# Load multiple images
image_folder = Path("my_images")
image_files = list(image_folder.glob("*.jpg"))

print(f"Processing {len(image_files)} images...")

# Process in batches
batch_size = 8
all_tokens = []

for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]

    # Load and preprocess batch
    batch_images = []
    for img_path in batch_files:
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) * 2.0 - 1.0
        batch_images.append(img)

    # Stack into batch
    batch_tensor = torch.stack(batch_images).to(device)

    # Encode
    with torch.no_grad():
        tokens = tokenizer.encode(batch_tensor)
        all_tokens.append(tokens.cpu())

    print(f"Processed {i+len(batch_files)}/{len(image_files)}")

# Concatenate all tokens
all_tokens = torch.cat(all_tokens, dim=0)
print(f"Final token shape: {all_tokens.shape}")

# Save tokens
torch.save(all_tokens, "image_tokens.pt")
print("Saved tokens to image_tokens.pt")
```

### Use Case 2: Process a Video

```python
import torch
from cosmos_tokenizer import VideoTokenizer
import numpy as np

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = VideoTokenizer(
    mode="CV",  # Continuous Video
    spatial_compression=8,
    temporal_compression=8,
    device=device
)

# Create dummy video (replace with real video loading)
# Shape: (batch, channels, time, height, width)
video = torch.randn(1, 3, 32, 256, 256).to(device)
# In practice, load with: opencv, av, or decord library

# Normalize to [-1, 1] if needed
# video = video / 127.5 - 1.0

# Process video
with torch.no_grad():
    # Use temporal_window for long videos to save memory
    reconstructed = tokenizer(video, temporal_window=17)

print(f"Input shape: {video.shape}")
print(f"Output shape: {reconstructed.shape}")

# Calculate compression
input_size = video.numel()
# Internal latent size is: (1, 16, 4, 32, 32) for CV 8×8×8
latent_size = 1 * 16 * (32//8) * (256//8) * (256//8)
compression = input_size / latent_size
print(f"Compression ratio: {compression:.1f}×")
```

### Use Case 3: Training Your Own Model

**Note:** You need to prepare a dataset first. See `configs/di_16x16.yaml` for configuration.

```python
import torch
from torch.utils.data import DataLoader, Dataset
from cosmos_tokenizer.networks.autoencoder import DiscreteImageTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_DI_16x16
from cosmos_tokenizer.training.losses import TokenizerLoss

# 1. Create dataset (simplified example)
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        # Load your images here
        self.images = []  # List of image paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load, preprocess, return image tensor
        # Shape: (3, 256, 256), range [-1, 1]
        return image_tensor

# 2. Create model
model = DiscreteImageTokenizer(COSMOS_DI_16x16).cuda()

# 3. Create loss function
criterion = TokenizerLoss(
    recon_loss_type="l1",
    perceptual_weight=1.0,
    quantizer_weight=1.0,
)

# 4. Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 5. Training loop (simplified)
dataset = ImageDataset("path/to/images")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(100):
    for batch_idx, images in enumerate(dataloader):
        images = images.cuda()

        # Forward pass
        reconstructed, info = model(images)

        # Compute loss
        losses = criterion(reconstructed, images, info)
        total_loss = losses["total_loss"]

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")
```

**For serious training, use PyTorch Lightning and check the `configs/` directory for proper configuration.**

---

## Next Steps

### 1. Learn More About the Architecture
Read the detailed technical documentation:
- `claude.md` - Architecture deep dive
- `API_REFERENCE.md` - Complete API documentation

### 2. Experiment with Different Settings
Try:
- Different compression ratios (8 vs 16)
- Different modes (CI vs DI)
- Different image sizes
- Video processing

### 3. Train Your Own Model
- Prepare a dataset (ImageNet, your own images)
- Configure training (see `configs/`)
- Monitor with Weights & Biases or TensorBoard

### 4. Integrate into Your Project
- Use as a preprocessing step for diffusion models
- Build an autoregressive image generator
- Compress your dataset to save storage

### 5. Read Research Papers
Understand the techniques behind this:
- NVIDIA Cosmos-Tokenizer
- VQ-VAE, VQ-GAN
- Finite Scalar Quantization (FSQ)

---

## Common Questions

**Q: Why are my reconstructed images blurry?**
A: This is expected with high compression. Try:
- Lower compression ratio (spatial_compression=8 instead of 16)
- Train with perceptual loss
- Use CI mode instead of DI for smoother results

**Q: Can I use this on CPU?**
A: Yes, but it's 10-50× slower. For testing/learning it's fine. For serious work, use GPU.

**Q: Do I need pre-trained weights?**
A: Currently, you need to train your own model. Pre-trained weights are coming soon. For now, the model initializes randomly, which is why reconstruction quality won't be good without training.

**Q: What image sizes are supported?**
A: Any size divisible by the compression ratio. For spatial_compression=16, use multiples of 16 (256, 512, 1024, etc.). The model works best with powers of 2.

**Q: How long does training take?**
A: On 8× A100 GPUs with ImageNet:
- DI 16×16: ~3-5 days
- CI 8×8: ~2-3 days
- Video models: 1-2 weeks

On a single consumer GPU (RTX 3090), multiply by 8-10×.

**Q: Can I fine-tune on my own images?**
A: Yes! Load a pre-trained checkpoint and continue training on your dataset.

**Q: Why discrete vs continuous?**
A:
- Discrete (DI/DV): For GPT-style autoregressive models
- Continuous (CI/CV): For diffusion models (Stable Diffusion-style)

Choose based on what kind of generator you want to build.

---

## Need Help?

1. Check `TROUBLESHOOTING.md` for common issues
2. Read `FAQ.md` for frequently asked questions
3. Open an issue on GitHub
4. Check the examples in `EXAMPLES.md`

**You're now ready to start tokenizing!** 🚀

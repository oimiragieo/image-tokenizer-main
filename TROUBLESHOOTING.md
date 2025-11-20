# Troubleshooting Guide

Common issues and solutions for Cosmos Image Tokenizer.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [CUDA/GPU Issues](#cudagpu-issues)
- [Memory Issues](#memory-issues)
- [Quality/Output Issues](#qualityoutput-issues)
- [Training Issues](#training-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### Problem: `pip install -e .` fails with "No module named 'torch'

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
Install PyTorch first before installing the tokenizer:

```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision

# Then install tokenizer
pip install -e .
```

**Why:** PyTorch is listed as a dependency but pip doesn't always install it from the right source (especially for CUDA versions).

---

### Problem: `ImportError: cannot import name 'einx'`

**Error:**
```
ImportError: cannot import name 'einx' from 'einops'
```

**Solution:**
Install `einx` separately (it's different from `einops`):

```bash
pip install einx>=0.1.3
```

**Why:** `einx` is a separate package from `einops`, not a submodule.

---

### Problem: Installation succeeds but `cosmos-image` command not found

**Error:**
```bash
cosmos-image encode ...
bash: cosmos-image: command not found
```

**Solution:**
The CLI tools are not yet implemented. Use the Python API instead:

```python
from cosmos_tokenizer import ImageTokenizer

tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
# ... use Python API
```

**Status:** CLI tools are planned but not yet available (see `setup.py` entry points).

---

## Import Errors

### Problem: `ModuleNotFoundError: No module named 'cosmos_tokenizer'`

**Symptoms:**
```python
from cosmos_tokenizer import ImageTokenizer
# ModuleNotFoundError: No module named 'cosmos_tokenizer'
```

**Solutions:**

1. **Check installation:**
   ```bash
   pip list | grep cosmos
   # Should show: cosmos-tokenizer  1.0.0
   ```

2. **Reinstall in development mode:**
   ```bash
   cd /path/to/cosmos-image-tokenizer
   pip install -e .
   ```

3. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   # Should include /path/to/cosmos-image-tokenizer
   ```

4. **Are you in the right environment?**
   ```bash
   which python
   # Make sure it's from your virtual environment
   ```

---

### Problem: `ImportError: cannot import name 'ImageTokenizer'`

**Error:**
```python
from cosmos_tokenizer import ImageTokenizer
# ImportError: cannot import name 'ImageTokenizer' from 'cosmos_tokenizer'
```

**Solution:**
Check that all dependencies are installed:

```bash
pip install -e ".[all]"
```

Or check for missing modules:
```python
import cosmos_tokenizer
print(dir(cosmos_tokenizer))
# Should show: ['ImageTokenizer', 'VideoTokenizer', ...]
```

---

## CUDA/GPU Issues

### Problem: "CUDA out of memory" error

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB (GPU 0; Y GB total capacity)
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   # Instead of:
   batch = torch.randn(64, 3, 256, 256)  # Too large

   # Use:
   batch = torch.randn(8, 3, 256, 256)   # Smaller batch
   ```

2. **Use smaller compression:**
   ```python
   # Instead of spatial_compression=8 (uses more memory)
   # Use spatial_compression=16 (uses less memory)
   tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
   ```

3. **Use mixed precision:**
   ```python
   tokenizer = ImageTokenizer(
       mode="DI",
       spatial_compression=16,
       dtype="bfloat16"  # Uses half the memory
   )
   ```

4. **Process videos with temporal_window:**
   ```python
   # Instead of:
   reconstructed = tokenizer(long_video)

   # Use:
   reconstructed = tokenizer(long_video, temporal_window=17)
   ```

5. **Clear cache between batches:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

6. **Check GPU memory:**
   ```bash
   nvidia-smi
   # Shows current GPU memory usage
   ```

---

### Problem: Model runs on CPU instead of GPU

**Symptoms:**
- Very slow processing (10-50× slower than expected)
- `nvidia-smi` shows 0% GPU utilization

**Check:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**Solutions:**

1. **PyTorch installed without CUDA:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # Should print: 11.8 or 12.1, NOT None
   ```

   If it prints `None`, reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Explicitly set device:**
   ```python
   tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")
   ```

3. **Check CUDA driver:**
   ```bash
   nvidia-smi
   # Should show driver version and GPUs
   ```

   If this fails, update NVIDIA drivers.

---

### Problem: `RuntimeError: CUDA error: device-side assert triggered`

**Causes:**
- Invalid tensor values (NaN, Inf)
- Out-of-bounds indices
- Incorrect input range

**Solutions:**

1. **Check for NaN/Inf:**
   ```python
   image = ...
   assert not torch.isnan(image).any(), "Image contains NaN"
   assert not torch.isinf(image).any(), "Image contains Inf"
   ```

2. **Check input range:**
   ```python
   # Images must be in [-1, 1]
   assert image.min() >= -1.0 and image.max() <= 1.0
   ```

3. **Run on CPU for better error messages:**
   ```python
   tokenizer = ImageTokenizer(mode="DI", device="cpu")
   # Error message will be more informative
   ```

---

## Memory Issues

### Problem: Python process killed / "Killed" message

**Symptoms:**
```bash
python my_script.py
Killed
```

**Cause:** Out of system RAM (not GPU memory)

**Solutions:**

1. **Check system memory:**
   ```bash
   free -h
   # Watch available memory
   ```

2. **Reduce batch size**

3. **Use data generators instead of loading all at once:**
   ```python
   # Bad:
   all_images = [load_image(f) for f in files]  # Loads everything

   # Good:
   for file in files:
       image = load_image(file)
       tokens = tokenizer.encode(image)
       # Process one at a time
   ```

4. **Enable swap (Linux):**
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Quality/Output Issues

### Problem: Reconstructed images are very blurry or low quality

**Cause:** Model has random weights (not trained)

**Explanation:**
The library provides the architecture but **not pre-trained weights**. Without training, the model produces poor reconstructions.

**Solutions:**

1. **Train your own model** (see `configs/` and training examples)

2. **Wait for pre-trained checkpoints** (coming soon)

3. **Use lower compression for better quality:**
   ```python
   # Better quality (but larger latents):
   tokenizer = ImageTokenizer(mode="DI", spatial_compression=8)

   # vs worse quality (but smaller latents):
   tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
   ```

---

### Problem: Reconstructed image range is wrong

**Symptoms:**
- All black or all white images
- Values outside expected range

**Check:**
```python
image = torch.randn(1, 3, 256, 256) * 2 - 1  # [-1, 1]
reconstructed = tokenizer(image)
print(f"Input range: [{image.min():.2f}, {image.max():.2f}]")
print(f"Output range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
# Both should be approximately [-1, 1]
```

**Solution:**
Ensure input is in correct range `[-1, 1]`:

```python
# If your image is in [0, 255]:
image = image / 255.0      # -> [0, 1]
image = image * 2.0 - 1.0  # -> [-1, 1]

# When converting back:
output = (reconstructed + 1.0) / 2.0  # -> [0, 1]
output = torch.clamp(output, 0, 1)
output = (output * 255).byte()        # -> [0, 255]
```

---

### Problem: Image dimensions are wrong after reconstruction

**Error:**
```
RuntimeError: Expected 256x256 but got 252x252
```

**Cause:** Input size not divisible by compression factor × patch size

**Solution:**
Use sizes divisible by `spatial_compression * patch_size`:

```python
# For spatial_compression=16, patch_size=4:
# 16 * 4 = 64, so use multiples of 64
valid_sizes = [64, 128, 192, 256, 320, 384, 448, 512, ...]

# Resize your image:
from PIL import Image
img = Image.open("photo.jpg")
img = img.resize((256, 256))  # or (512, 512), etc.
```

**Recommended sizes:** 256, 512, 1024 (powers of 2)

---

## Training Issues

### Problem: Loss is NaN or exploding

**Error:**
```
Epoch 1, Batch 100, Loss: nan
```

**Solutions:**

1. **Check learning rate (might be too high):**
   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   # Try: lr=1e-5 or lr=5e-5
   ```

2. **Clip gradients:**
   ```python
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```

3. **Check for NaN in data:**
   ```python
   assert not torch.isnan(images).any()
   assert not torch.isinf(images).any()
   ```

4. **Use gradient scaling for mixed precision:**
   ```python
   from torch.cuda.amp import GradScaler
   scaler = GradScaler()

   optimizer.zero_grad()
   with autocast():
       loss = ...
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

---

### Problem: Training is very slow

**Symptoms:**
- <1 iteration per second
- GPU utilization <50%

**Solutions:**

1. **Use mixed precision:**
   ```python
   from torch.cuda.amp import autocast

   with autocast():
       reconstruction, info = model(images)
       loss = criterion(reconstruction, images, info)["total_loss"]
   ```

2. **Increase batch size (if memory allows)**

3. **Use more dataloader workers:**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=8,  # Increase this
       pin_memory=True,
       persistent_workers=True
   )
   ```

4. **Pre-load data to faster storage** (SSD instead of HDD)

5. **Disable perceptual loss during early training:**
   ```python
   # Perceptual loss is slow (uses VGG)
   criterion = TokenizerLoss(
       use_perceptual=False,  # Disable for first few epochs
       recon_loss_type="l1"
   )
   ```

---

### Problem: Codebook collapse (discrete models)

**Symptoms:**
- Perplexity very low (<100)
- Model uses only a small fraction of codebook
- Poor reconstruction quality

**Check:**
```python
_, info = model(images)
print(f"Perplexity: {info['perplexity']}")
# Should be >1000 for healthy codebook usage
```

**Solutions:**

1. **Increase codebook usage penalty:**
   ```python
   # For VQ:
   quantizer = VectorQuantizer(
       ...,
       commitment_cost=0.5,  # Increase from 0.25
   )
   ```

2. **Use FSQ instead of VQ** (less prone to collapse):
   ```python
   config.quantizer_type = "fsq"
   ```

3. **Reduce learning rate**

4. **Train for longer** (collapse sometimes resolves)

---

## Performance Issues

### Problem: Inference is slow (lower FPS than documented)

**Benchmarks claim 1000+ FPS but you get <100 FPS**

**Solutions:**

1. **Use mixed precision:**
   ```python
   tokenizer = ImageTokenizer(dtype="bfloat16")
   # Or:
   tokenizer = ImageTokenizer(dtype="float16")
   ```

2. **Use torch.compile (PyTorch 2.0+):**
   ```python
   tokenizer.model = torch.compile(tokenizer.model)
   ```

3. **Use TorchScript:**
   ```python
   jit_model = tokenizer.to_jit()
   torch.jit.save(jit_model, "tokenizer.jit")
   ```

4. **Disable autograd:**
   ```python
   with torch.no_grad():  # Essential for inference
       tokens = tokenizer.encode(image)
   ```

5. **Batch images:**
   ```python
   # Slower:
   for img in images:
       tokens = tokenizer.encode(img.unsqueeze(0))

   # Faster:
   batch = torch.stack(images)
   tokens = tokenizer.encode(batch)
   ```

6. **Use GPU (not CPU)**

---

### Problem: Video processing runs out of memory

**Error:**
```
CUDA out of memory (processing 256x256x128 video)
```

**Solution:**
Use `temporal_window` parameter:

```python
# Instead of:
reconstructed = tokenizer(long_video)  # OOM!

# Use:
reconstructed = tokenizer(long_video, temporal_window=17)
```

**How it works:** Processes video in overlapping 17-frame windows, much more memory-efficient.

**Recommended window sizes:**
- 17 frames: Good balance
- 9 frames: Very memory-constrained
- 33 frames: If you have lots of memory

---

## Error Message Decoder

### `RuntimeError: Expected all tensors to be on the same device`

**Solution:**
```python
image = image.to(tokenizer.device)
# Or initialize with correct device:
tokenizer = ImageTokenizer(device="cuda")
```

### `ValueError: Invalid mode for ImageTokenizer: CV`

**Solution:**
`CV` is for VideoTokenizer, not ImageTokenizer:
```python
# Wrong:
tokenizer = ImageTokenizer(mode="CV")

# Right:
tokenizer = VideoTokenizer(mode="CV")
```

### `AssertionError: CausalVideoTokenizer requires causal=True in config`

**Solution:**
Use `VideoTokenizer` instead, or ensure config has `causal=True`:
```python
# Option 1:
tokenizer = VideoTokenizer(mode="DV")

# Option 2:
config = TokenizerConfig(mode="DV", causal=True, ...)
tokenizer = CausalVideoTokenizer(config=config)
```

### `TypeError: forward() got an unexpected keyword argument 'temporal_window'`

**Solution:**
`temporal_window` is only for VideoTokenizer:
```python
# Wrong:
image_tokenizer(image, temporal_window=17)

# Right:
video_tokenizer(video, temporal_window=17)
```

---

## Getting Help

If you're still stuck:

1. **Check the FAQ:** `FAQ.md`
2. **Review examples:** `EXAMPLES.md`
3. **Search existing issues:** GitHub Issues
4. **Ask for help:** Open a new issue with:
   - Full error message
   - Minimal code to reproduce
   - PyTorch/CUDA versions (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
   - GPU info (`nvidia-smi`)

---

## Debug Checklist

When something doesn't work:

- [ ] Python >= 3.10?
- [ ] PyTorch >= 2.0.0?
- [ ] CUDA available? (`torch.cuda.is_available()`)
- [ ] Installed with `pip install -e .`?
- [ ] Input in range [-1, 1]?
- [ ] Input shape correct? (B, 3, H, W) for images
- [ ] Using correct mode? (CI/DI for images, CV/DV for videos)
- [ ] GPU has enough memory?
- [ ] Using `torch.no_grad()` for inference?
- [ ] Model on same device as data?
- [ ] Latest version of dependencies?

Run this diagnostic script:

```python
import torch
from cosmos_tokenizer import ImageTokenizer

print("=== Diagnostic Info ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    tokenizer = ImageTokenizer(mode="DI", device="cuda" if torch.cuda.is_available() else "cpu")
    print("✓ Tokenizer created successfully")

    image = torch.randn(1, 3, 256, 256) * 2 - 1
    image = image.to(tokenizer.device)
    with torch.no_grad():
        tokens = tokenizer.encode(image)
        reconstructed = tokenizer.decode(tokens)
    print("✓ Encode/decode successful")
    print(f"  Input shape: {image.shape}")
    print(f"  Token shape: {tokens.shape}")
    print(f"  Output shape: {reconstructed.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
```

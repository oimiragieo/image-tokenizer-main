# Code Examples

Practical, copy-paste examples for common use cases.

## Table of Contents
- [Basic Image Tokenization](#basic-image-tokenization)
- [Batch Processing](#batch-processing)
- [Video Tokenization](#video-tokenization)
- [Working with Real Images](#working-with-real-images)
- [Training Examples](#training-examples)
- [Integration Examples](#integration-examples)
- [Performance Optimization](#performance-optimization)

---

## Basic Image Tokenization

### Example 1: Simplest Possible Usage

```python
import torch
from cosmos_tokenizer import ImageTokenizer

# Create tokenizer
tokenizer = ImageTokenizer(
    mode="DI",  # Discrete Image
    spatial_compression=16,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create random image (B, C, H, W), range [-1, 1]
image = torch.randn(1, 3, 256, 256) * 2 - 1

# Encode to tokens
tokens = tokenizer.encode(image)
print(f"Token shape: {tokens.shape}")  # (1, 16, 16)

# Decode back to image
reconstructed = tokenizer.decode(tokens)
print(f"Reconstructed shape: {reconstructed.shape}")  # (1, 3, 256, 256)
```

---

### Example 2: Continuous Mode (for Diffusion Models)

```python
import torch
from cosmos_tokenizer import ImageTokenizer

# Create continuous tokenizer
tokenizer = ImageTokenizer(
    mode="CI",  # Continuous Image
    spatial_compression=8,
    device="cuda"
)

# Generate image
image = torch.randn(1, 3, 256, 256) * 2 - 1

# Encode to continuous latent
latent = tokenizer.encode(image, deterministic=True)
print(f"Latent shape: {latent.shape}")  # (1, 16, 32, 32)
print(f"Latent type: continuous (floats)")

# Decode
reconstructed = tokenizer.decode(latent)

# For diffusion models, you'd train on these latents
```

---

### Example 3: Get Detailed Information

```python
# Get full output with metadata
output = tokenizer.encode(image, return_dict=True)

print(f"Latent: {output.latent.shape}")

if hasattr(output, 'indices'):
    print(f"Indices: {output.indices.shape}")
    print(f"Unique tokens used: {output.indices.unique().numel()}")

if hasattr(output, 'perplexity'):
    print(f"Codebook perplexity: {output.perplexity:.1f}")
    print(f"(Higher is better, indicates diverse codebook usage)")
```

---

## Batch Processing

### Example 4: Process Multiple Images Efficiently

```python
import torch
from cosmos_tokenizer import ImageTokenizer

tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")

# Batch of 16 images
batch = torch.randn(16, 3, 256, 256) * 2 - 1
batch = batch.cuda()

# Process entire batch at once (faster than loop)
with torch.no_grad():
    tokens = tokenizer.encode(batch)  # (16, 16, 16)
    reconstructed = tokenizer.decode(tokens)  # (16, 3, 256, 256)

print(f"Processed {batch.shape[0]} images in one batch")
```

---

### Example 5: Process Image Directory

```python
import torch
from cosmos_tokenizer import ImageTokenizer
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_and_preprocess(image_path, size=256):
    """Load image and convert to tensor."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]
    return img_tensor

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device=device)

# Get all images
image_dir = Path("path/to/images")
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

# Process in batches
batch_size = 16
all_tokens = []

for i in tqdm(range(0, len(image_files), batch_size), desc="Tokenizing"):
    # Load batch
    batch_files = image_files[i:i+batch_size]
    batch_images = [load_and_preprocess(f) for f in batch_files]
    batch_tensor = torch.stack(batch_images).to(device)

    # Tokenize
    with torch.no_grad():
        tokens = tokenizer.encode(batch_tensor)
        all_tokens.append(tokens.cpu())

# Concatenate all
all_tokens = torch.cat(all_tokens, dim=0)
print(f"Tokenized {all_tokens.shape[0]} images")
print(f"Token tensor shape: {all_tokens.shape}")

# Save
torch.save({
    'tokens': all_tokens,
    'filenames': [str(f) for f in image_files],
    'config': {
        'mode': 'DI',
        'spatial_compression': 16,
    }
}, "tokenized_dataset.pt")

print("Saved to tokenized_dataset.pt")
```

---

## Video Tokenization

### Example 6: Basic Video Tokenization

```python
import torch
from cosmos_tokenizer import VideoTokenizer

# Create video tokenizer
tokenizer = VideoTokenizer(
    mode="CV",  # Continuous Video
    spatial_compression=8,
    temporal_compression=8,
    device="cuda"
)

# Video: (B, C, T, H, W)
video = torch.randn(1, 3, 32, 256, 256) * 2 - 1
video = video.cuda()

# Encode
with torch.no_grad():
    latent = tokenizer.encode(video)

print(f"Video shape: {video.shape}")  # (1, 3, 32, 256, 256)
print(f"Latent shape: {latent.shape}")  # (1, 16, 4, 32, 32)
# 32 frames → 4 temporal tokens (32/8)
# 256×256 → 32×32 spatial tokens (256/8)

# Decode
with torch.no_grad():
    reconstructed = tokenizer.decode(latent)

print(f"Reconstructed shape: {reconstructed.shape}")  # (1, 3, 32, 256, 256)
```

---

### Example 7: Long Video with Sliding Window

```python
import torch
from cosmos_tokenizer import VideoTokenizer

tokenizer = VideoTokenizer(
    mode="CV",
    spatial_compression=8,
    temporal_compression=8,
    device="cuda"
)

# Long video (128 frames)
long_video = torch.randn(1, 3, 128, 256, 256) * 2 - 1
long_video = long_video.cuda()

# Process with temporal window (memory-efficient)
with torch.no_grad():
    reconstructed = tokenizer(
        long_video,
        temporal_window=17  # Process 17 frames at a time
    )

print(f"Processed {long_video.shape[2]} frames using sliding window")
print(f"Peak memory usage was for ~17 frames, not all 128")
```

---

### Example 8: Load Video from File

```python
import torch
import cv2
import numpy as np
from cosmos_tokenizer import VideoTokenizer

def load_video(video_path, num_frames=32, size=256):
    """Load video from file."""
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
        frame_count += 1

    cap.release()

    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1])  # Repeat last frame

    # Convert to tensor
    video_array = np.array(frames).astype(np.float32) / 255.0  # [0, 1]
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)  # (T, 3, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (3, T, H, W)
    video_tensor = video_tensor * 2.0 - 1.0  # [-1, 1]
    video_tensor = video_tensor.unsqueeze(0)  # (1, 3, T, H, W)

    return video_tensor

# Load and tokenize
tokenizer = VideoTokenizer(mode="CV", spatial_compression=8, temporal_compression=8, device="cuda")
video = load_video("video.mp4", num_frames=32, size=256).cuda()

with torch.no_grad():
    reconstructed = tokenizer(video, temporal_window=17)

print(f"Loaded and tokenized video: {video.shape}")
```

---

## Working with Real Images

### Example 9: Load, Tokenize, and Save Images

```python
import torch
from cosmos_tokenizer import ImageTokenizer
from PIL import Image
import numpy as np

def image_to_tensor(pil_image):
    """Convert PIL Image to tensor in [-1, 1]."""
    # Ensure RGB
    pil_image = pil_image.convert("RGB")

    # To numpy array [0, 255]
    img_array = np.array(pil_image).astype(np.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # To tensor (H, W, C) → (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

    # Scale to [-1, 1]
    img_tensor = img_tensor * 2.0 - 1.0

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def tensor_to_image(tensor):
    """Convert tensor in [-1, 1] to PIL Image."""
    # Remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move to CPU
    tensor = tensor.cpu()

    # Scale to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # To numpy (C, H, W) → (H, W, C)
    img_array = tensor.permute(1, 2, 0).numpy()

    # Scale to [0, 255]
    img_array = (img_array * 255).astype(np.uint8)

    # To PIL
    pil_image = Image.fromarray(img_array)

    return pil_image

# Load image
pil_img = Image.open("photo.jpg")
pil_img = pil_img.resize((256, 256))

# Convert to tensor
img_tensor = image_to_tensor(pil_img)

# Tokenize
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")
img_tensor = img_tensor.cuda()

with torch.no_grad():
    tokens = tokenizer.encode(img_tensor)
    reconstructed = tokenizer.decode(tokens)

# Convert back to PIL and save
reconstructed_pil = tensor_to_image(reconstructed)
reconstructed_pil.save("reconstructed.jpg")

print("Saved reconstructed image")

# Calculate and save tokens
torch.save(tokens.cpu(), "image_tokens.pt")
print(f"Saved tokens: {tokens.shape}")
```

---

### Example 10: Compare Original vs Reconstructed

```python
import torch
from cosmos_tokenizer import ImageTokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ... (use helper functions from Example 9) ...

# Load and process
original_pil = Image.open("photo.jpg").resize((256, 256))
img_tensor = image_to_tensor(original_pil).cuda()

# Tokenize
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")

with torch.no_grad():
    tokens = tokenizer.encode(img_tensor)
    reconstructed_tensor = tokenizer.decode(tokens)

reconstructed_pil = tensor_to_image(reconstructed_tensor)

# Calculate metrics
img_np = np.array(original_pil).astype(np.float32)
rec_np = np.array(reconstructed_pil).astype(np.float32)

mse = np.mean((img_np - rec_np) ** 2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_pil)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(reconstructed_pil)
axes[1].set_title(f"Reconstructed (PSNR: {psnr:.1f} dB)")
axes[1].axis('off')

# Difference
diff = np.abs(img_np - rec_np).astype(np.uint8)
axes[2].imshow(diff)
axes[2].set_title("Absolute Difference (amplified)")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
print("Saved comparison.png")
```

---

## Training Examples

### Example 11: Simple Training Loop

```python
import torch
from torch.utils.data import Dataset, DataLoader
from cosmos_tokenizer.networks.autoencoder import DiscreteImageTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_DI_16x16
from cosmos_tokenizer.training.losses import TokenizerLoss
from tqdm import tqdm

# Create dataset (simplified)
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image, preprocess, return tensor
        # (Implementation depends on your data format)
        img = load_and_preprocess(self.image_paths[idx])  # Returns (3, 256, 256) in [-1, 1]
        return img

# Setup
device = "cuda"
model = DiscreteImageTokenizer(COSMOS_DI_16x16).to(device)

# Loss function
criterion = TokenizerLoss(
    recon_loss_type="l1",
    perceptual_weight=1.0,
    quantizer_weight=1.0,
    use_perceptual=True,
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Dataloader
dataset = ImageDataset(your_image_paths)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)

        # Forward pass
        reconstructed, info = model(images)

        # Compute loss
        losses = criterion(reconstructed, images, info)
        total_loss = losses["total_loss"]

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log
        epoch_loss += total_loss.item()
        pbar.set_postfix({
            'loss': total_loss.item(),
            'recon': losses['recon_loss'].item(),
            'perceptual': losses.get('perceptual_loss', 0),
        })

    # Epoch summary
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"checkpoint_epoch{epoch+1}.pt")

print("Training complete!")
```

---

### Example 12: Training with Mixed Precision

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Setup (same as Example 11)
model = DiscreteImageTokenizer(COSMOS_DI_16x16).to("cuda")
criterion = TokenizerLoss(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Gradient scaler for mixed precision
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    for images in dataloader:
        images = images.cuda()

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            reconstructed, info = model(images)
            losses = criterion(reconstructed, images, info)
            total_loss = losses["total_loss"]

        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

print("Training with mixed precision - 2-3× faster!")
```

---

## Integration Examples

### Example 13: Use as Preprocessing for Diffusion Model

```python
import torch
from cosmos_tokenizer import ImageTokenizer

# Create continuous tokenizer
tokenizer = ImageTokenizer(mode="CI", spatial_compression=8, device="cuda")

# Preprocess entire dataset to latents
def create_latent_dataset(image_dataloader, output_path):
    all_latents = []

    for images in tqdm(image_dataloader):
        images = images.cuda()
        with torch.no_grad():
            latents = tokenizer.encode(images, deterministic=True)
            all_latents.append(latents.cpu())

    # Save
    latents_tensor = torch.cat(all_latents, dim=0)
    torch.save(latents_tensor, output_path)
    print(f"Saved {latents_tensor.shape[0]} latents to {output_path}")

# Use in diffusion model training
class LatentDataset(Dataset):
    def __init__(self, latent_file):
        self.latents = torch.load(latent_file)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]

# Train diffusion model on latents (not images!)
latent_dataset = LatentDataset("latents.pt")
# ... train diffusion model ...
```

---

### Example 14: Build GPT-style Image Generator

```python
import torch
import torch.nn as nn
from cosmos_tokenizer import ImageTokenizer

# Step 1: Tokenize images to discrete tokens
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")

# Step 2: Build autoregressive model
class ImageGPT(nn.Module):
    def __init__(self, vocab_size=64000, seq_len=256, d_model=512, n_layers=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=n_layers
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        # tokens: (B, seq_len)
        x = self.token_emb(tokens) + self.pos_emb[:, :tokens.size(1)]
        x = self.transformer(x)
        logits = self.output_proj(x)
        return logits

# Step 3: Prepare data
images = load_images()  # (B, 3, 256, 256)
with torch.no_grad():
    tokens = tokenizer.encode(images)  # (B, 16, 16)
    tokens_flat = tokens.flatten(1)  # (B, 256)

# Step 4: Train GPT
gpt = ImageGPT(vocab_size=64000, seq_len=256).cuda()
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_tokens in dataloader:
        # Shift for autoregressive training
        input_tokens = batch_tokens[:, :-1]  # All but last
        target_tokens = batch_tokens[:, 1:]  # All but first

        # Forward
        logits = gpt(input_tokens)

        # Loss
        loss = criterion(
            logits.reshape(-1, 64000),
            target_tokens.reshape(-1)
        )

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 5: Generate new images
def generate_image(gpt, tokenizer, start_token=0):
    tokens = torch.tensor([[start_token]]).cuda()

    for _ in range(255):  # Generate 256 tokens total
        logits = gpt(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    # Reshape to 16×16
    tokens_2d = tokens.reshape(1, 16, 16)

    # Decode to image
    with torch.no_grad():
        image = tokenizer.decode(tokens_2d)

    return image

# Generate!
generated_image = generate_image(gpt, tokenizer)
```

---

## Performance Optimization

### Example 15: Benchmark Different Configurations

```python
import torch
import time
from cosmos_tokenizer import ImageTokenizer

def benchmark(mode, compression, dtype, num_iters=100):
    tokenizer = ImageTokenizer(
        mode=mode,
        spatial_compression=compression,
        dtype=dtype,
        device="cuda"
    )

    # Warmup
    dummy = torch.randn(8, 3, 256, 256).cuda() * 2 - 1
    for _ in range(10):
        with torch.no_grad():
            _ = tokenizer(dummy)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iters):
        batch = torch.randn(8, 3, 256, 256).cuda() * 2 - 1
        with torch.no_grad():
            _ = tokenizer(batch)

    torch.cuda.synchronize()
    end = time.time()

    total_images = num_iters * 8
    elapsed = end - start
    fps = total_images / elapsed

    return fps

# Test different configurations
configs = [
    ("DI", 16, "float32"),
    ("DI", 16, "bfloat16"),
    ("DI", 8, "float32"),
    ("CI", 16, "float32"),
]

print("Configuration Benchmark:")
print(f"{'Mode':<5} {'Compression':<12} {'Dtype':<10} {'FPS':<10}")
print("-" * 50)

for mode, compression, dtype in configs:
    fps = benchmark(mode, compression, dtype)
    print(f"{mode:<5} {compression:<12} {dtype:<10} {fps:>8.1f}")
```

---

### Example 16: Distributed Training (Multi-GPU)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed():
    # Setup
    local_rank = setup_distributed()

    # Create model
    model = DiscreteImageTokenizer(COSMOS_DI_16x16).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create dataset with distributed sampler
    dataset = ImageDataset(...)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for shuffling

        for images in dataloader:
            images = images.cuda(local_rank)

            # Forward
            reconstructed, info = model(images)

            # Loss and backward
            loss = criterion(reconstructed, images, info)["total_loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Only log from rank 0
            if local_rank == 0:
                print(f"Loss: {loss.item()}")

    dist.destroy_process_group()

# Run with:
# torchrun --nproc_per_node=8 train_script.py
```

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Beginner tutorial
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [FAQ](FAQ.md) - Frequently asked questions

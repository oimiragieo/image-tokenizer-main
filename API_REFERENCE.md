# Cosmos Tokenizer API Reference

Complete API documentation for Cosmos Image Tokenizer.

## Table of Contents
- [ImageTokenizer](#imagetokenizer)
- [VideoTokenizer](#videotokenizer)
- [CausalVideoTokenizer](#causalvideotokenizer)
- [TokenizerConfig](#tokenizerconfig)
- [Quantizers](#quantizers)
- [Network Classes](#network-classes)
- [Loss Functions](#loss-functions)

---

## ImageTokenizer

High-level unified API for image tokenization supporting both continuous (CI) and discrete (DI) modes.

### Class Signature

```python
class ImageTokenizer(nn.Module):
    def __init__(
        self,
        mode: Literal["CI", "DI"] = "DI",
        spatial_compression: int = 16,
        config: Optional[TokenizerConfig] = None,
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "float32",
    )
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `"CI"` or `"DI"` | `"DI"` | Tokenizer mode. `"CI"` for continuous (VAE-style), `"DI"` for discrete (VQ-style) |
| `spatial_compression` | `int` | `16` | Spatial compression factor. Options: `8` or `16` |
| `config` | `TokenizerConfig` or `None` | `None` | Custom configuration object. If provided, overrides `mode` and `spatial_compression` |
| `checkpoint` | `str` or `None` | `None` | Path to pre-trained checkpoint file (.pt, .pth, or .jit) |
| `device` | `str` or `torch.device` | `"cuda"` | Device to run model on (`"cuda"`, `"cpu"`, or `torch.device`) |
| `dtype` | `str` or `torch.dtype` | `"float32"` | Data type for model weights. Options: `"float32"`, `"float16"`, `"bfloat16"` (or `"fp32"`, `"fp16"`, `"bf16"`) |

### Methods

#### `encode()`

Encode an image to latent representation.

```python
def encode(
    self,
    image: torch.Tensor,
    deterministic: bool = False,
    return_dict: bool = False,
) -> Union[torch.Tensor, dict]
```

**Parameters:**
- `image` (`torch.Tensor`): Input image of shape `(B, 3, H, W)` with values in range `[-1, 1]`
  - `B`: Batch size
  - `H`, `W`: Height and width (must be divisible by `spatial_compression * patch_size`)
  - Recommended sizes: 256, 512, 1024 (powers of 2)

- `deterministic` (`bool`, default=`False`):
  - For `CI` mode: If `True`, uses mean of distribution instead of sampling (no randomness)
  - For `DI` mode: No effect (always deterministic)

- `return_dict` (`bool`, default=`False`): If `True`, returns full `EncoderOutput` dict with additional info

**Returns:**
- For `DI` mode: `torch.Tensor` of shape `(B, h, w)` with integer indices in range `[0, 63999]`
  - `h = H / spatial_compression`, `w = W / spatial_compression`
  - Example: 256×256 image with `spatial_compression=16` → `16×16` tokens

- For `CI` mode: `torch.Tensor` of shape `(B, z_channels, h, w)` with continuous values
  - `z_channels`: Number of latent channels (16 for CI-8×8, 16 for CI-16×16)

- If `return_dict=True`: Dictionary with keys:
  - `latent`: The latent tensor
  - `indices`: Token indices (DI mode only)
  - `perplexity`: Codebook usage metric (DI mode only)

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
image = torch.randn(4, 3, 256, 256) * 2 - 1  # 4 images, range [-1, 1]

# Encode to discrete tokens
tokens = tokenizer.encode(image)
print(tokens.shape)  # torch.Size([4, 16, 16])
print(tokens.min(), tokens.max())  # 0, 63999

# Get full output
output = tokenizer.encode(image, return_dict=True)
print(output.keys())  # dict_keys(['latent', 'indices', 'perplexity'])
```

#### `decode()`

Decode latent representation back to image.

```python
def decode(
    self,
    latent: torch.Tensor,
    return_dict: bool = False,
) -> Union[torch.Tensor, dict]
```

**Parameters:**
- `latent` (`torch.Tensor`): Latent representation from `encode()`
  - For `DI` mode: Shape `(B, h, w)` with integer indices
  - For `CI` mode: Shape `(B, z_channels, h, w)` with continuous values

- `return_dict` (`bool`, default=`False`): If `True`, returns full `DecoderOutput` dict

**Returns:**
- `torch.Tensor`: Reconstructed image of shape `(B, 3, H, W)` with values in range `[-1, 1]`
- If `return_dict=True`: Dictionary with key `reconstruction`

**Example:**
```python
# Discrete mode
tokenizer_di = ImageTokenizer(mode="DI", spatial_compression=16)
tokens = tokenizer_di.encode(image)  # (B, 16, 16)
reconstructed = tokenizer_di.decode(tokens)  # (B, 3, 256, 256)

# Continuous mode
tokenizer_ci = ImageTokenizer(mode="CI", spatial_compression=8)
latent = tokenizer_ci.encode(image)  # (B, 16, 32, 32)
reconstructed = tokenizer_ci.decode(latent)  # (B, 3, 256, 256)
```

#### `forward()` / `autoencode()`

Full round-trip: encode then decode in one call.

```python
def forward(
    self,
    image: torch.Tensor,
    deterministic: bool = False,
) -> torch.Tensor
```

**Parameters:**
- `image` (`torch.Tensor`): Input image `(B, 3, H, W)`, range `[-1, 1]`
- `deterministic` (`bool`): For CI mode, use mean instead of sampling

**Returns:**
- `torch.Tensor`: Reconstructed image `(B, 3, H, W)`, range `[-1, 1]`

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
image = torch.randn(1, 3, 256, 256) * 2 - 1

# Shorthand for encode + decode
reconstructed = tokenizer(image)
# Equivalent to:
# tokens = tokenizer.encode(image)
# reconstructed = tokenizer.decode(tokens)

# Also available as:
reconstructed = tokenizer.autoencode(image)
```

#### `load_checkpoint()`

Load pre-trained weights from file.

```python
def load_checkpoint(self, checkpoint_path: str) -> None
```

**Parameters:**
- `checkpoint_path` (`str`): Path to checkpoint file

**Supported formats:**
- `.pt`, `.pth`: PyTorch state dict (may contain `"state_dict"` or `"model"` keys)
- `.jit`: TorchScript compiled model

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
tokenizer.load_checkpoint("checkpoints/di_16x16_epoch100.pt")

# Or pass during initialization
tokenizer = ImageTokenizer(
    mode="DI",
    spatial_compression=16,
    checkpoint="checkpoints/di_16x16_epoch100.pt"
)
```

#### `save_checkpoint()`

Save model weights to file.

```python
def save_checkpoint(self, checkpoint_path: str) -> None
```

**Parameters:**
- `checkpoint_path` (`str`): Path where to save checkpoint

**Saves:** Dictionary with keys `"state_dict"` and `"config"`

**Example:**
```python
tokenizer.save_checkpoint("my_tokenizer.pt")
```

#### `to_jit()`

Convert model to optimized TorchScript format.

```python
def to_jit(self) -> torch.jit.ScriptModule
```

**Returns:**
- `torch.jit.ScriptModule`: Compiled model for faster inference

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
jit_model = tokenizer.to_jit()
torch.jit.save(jit_model, "tokenizer.jit")

# Load later
loaded_jit = torch.jit.load("tokenizer.jit")
```

#### `get_codebook_size()`

Get vocabulary size for discrete tokenizers.

```python
def get_codebook_size(self) -> Optional[int]
```

**Returns:**
- `int`: Codebook size (64000 for DI mode)
- `None`: For CI mode (no codebook)

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
print(tokenizer.get_codebook_size())  # 64000
```

#### `get_compression_ratio()`

Get total spatial compression ratio.

```python
def get_compression_ratio(self) -> float
```

**Returns:**
- `float`: Compression ratio (e.g., 256.0 for 16×16 spatial compression)

**Example:**
```python
tokenizer = ImageTokenizer(mode="DI", spatial_compression=16)
print(tokenizer.get_compression_ratio())  # 256.0

# For spatial_compression=8:
# 8 * 8 = 64.0
```

---

## VideoTokenizer

High-level API for video tokenization supporting continuous (CV) and discrete (DV) modes with causal processing.

### Class Signature

```python
class VideoTokenizer(nn.Module):
    def __init__(
        self,
        mode: Literal["CV", "DV"] = "CV",
        spatial_compression: int = 8,
        temporal_compression: int = 8,
        config: Optional[TokenizerConfig] = None,
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "float32",
    )
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `"CV"` or `"DV"` | `"CV"` | Video tokenizer mode. `"CV"` for continuous, `"DV"` for discrete |
| `spatial_compression` | `int` | `8` | Spatial compression factor (8 or 16) |
| `temporal_compression` | `int` | `8` | Temporal compression factor (4 or 8) |
| `config` | `TokenizerConfig` or `None` | `None` | Custom configuration (overrides other params) |
| `checkpoint` | `str` or `None` | `None` | Path to checkpoint file |
| `device` | `str` or `torch.device` | `"cuda"` | Device (`"cuda"` or `"cpu"`) |
| `dtype` | `str` or `torch.dtype` | `"float32"` | Model dtype (`"float32"`, `"float16"`, `"bfloat16"`) |

### Methods

#### `encode()`

Encode video to latent representation.

```python
def encode(
    self,
    video: torch.Tensor,
    deterministic: bool = False,
    temporal_window: Optional[int] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, dict]
```

**Parameters:**
- `video` (`torch.Tensor`): Input video of shape `(B, 3, T, H, W)`, range `[-1, 1]`
  - `B`: Batch size
  - `T`: Number of frames
  - `H`, `W`: Height and width

- `deterministic` (`bool`): For CV mode, use mean instead of sampling

- `temporal_window` (`int` or `None`):
  - If provided and `T > temporal_window`, processes video in sliding windows
  - Recommended: `17` for long videos
  - Enables processing videos longer than GPU memory allows

- `return_dict` (`bool`): Return full output dict if `True`

**Returns:**
- For `DV` mode: Shape `(B, t, h, w)` with discrete indices
  - `t = T / temporal_compression`
  - `h = H / spatial_compression`, `w = W / spatial_compression`

- For `CV` mode: Shape `(B, z_channels, t, h, w)` with continuous values

**Example:**
```python
tokenizer = VideoTokenizer(
    mode="CV",
    spatial_compression=8,
    temporal_compression=8
)

# Short video (fits in memory)
video = torch.randn(1, 3, 32, 256, 256) * 2 - 1
latent = tokenizer.encode(video)
print(latent.shape)  # torch.Size([1, 16, 4, 32, 32])

# Long video (use sliding window)
long_video = torch.randn(1, 3, 128, 256, 256) * 2 - 1
latent = tokenizer.encode(long_video, temporal_window=17)
print(latent.shape)  # torch.Size([1, 16, 16, 32, 32])
```

#### `decode()`

Decode latent back to video.

```python
def decode(
    self,
    latent: torch.Tensor,
    temporal_window: Optional[int] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, dict]
```

**Parameters:**
- `latent` (`torch.Tensor`): Latent from `encode()`
- `temporal_window` (`int` or `None`): Use windowing for long sequences
- `return_dict` (`bool`): Return full output dict

**Returns:**
- `torch.Tensor`: Reconstructed video `(B, 3, T, H, W)`, range `[-1, 1]`

**Example:**
```python
latent = tokenizer.encode(video)
reconstructed = tokenizer.decode(latent)
```

#### `forward()` / `autoencode()`

Full round-trip for video.

```python
def forward(
    self,
    video: torch.Tensor,
    deterministic: bool = False,
    temporal_window: Optional[int] = None,
) -> torch.Tensor
```

**Parameters:**
- `video` (`torch.Tensor`): Input video `(B, 3, T, H, W)`, range `[-1, 1]`
- `deterministic` (`bool`): For CV mode, disable sampling
- `temporal_window` (`int` or `None`): Window size for long videos (recommended: 17)

**Returns:**
- `torch.Tensor`: Reconstructed video `(B, 3, T, H, W)`

**Example:**
```python
tokenizer = VideoTokenizer(mode="CV", spatial_compression=8, temporal_compression=8)
video = torch.randn(1, 3, 64, 256, 256) * 2 - 1

# Process with sliding window (recommended for >30 frames)
reconstructed = tokenizer(video, temporal_window=17)

# Equivalent to:
# latent = tokenizer.encode(video, temporal_window=17)
# reconstructed = tokenizer.decode(latent, temporal_window=17)
```

#### Checkpoint Methods

Same as `ImageTokenizer`:
- `load_checkpoint(checkpoint_path: str)`
- `save_checkpoint(checkpoint_path: str)`

---

## CausalVideoTokenizer

Specialized video tokenizer with explicit causal processing guarantee.

### Class Signature

```python
class CausalVideoTokenizer(VideoTokenizer):
    def __init__(self, *args, **kwargs)
```

**Note:** Inherits all methods from `VideoTokenizer`. Enforces `causal=True` in config.

**Use when:** You need guaranteed temporal causality (frame `t` cannot see frames `t+1`, `t+2`, ...). Essential for autoregressive video generation.

**Example:**
```python
from cosmos_tokenizer import CausalVideoTokenizer

tokenizer = CausalVideoTokenizer(
    mode="DV",  # Discrete Video
    spatial_compression=16,
    temporal_compression=8
)

# Process causally (each frame only sees past)
video = torch.randn(1, 3, 32, 256, 256) * 2 - 1
reconstructed = tokenizer(video)
```

---

## TokenizerConfig

Configuration dataclass for tokenizer models.

### Pre-defined Configs

Import from `cosmos_tokenizer.networks.configs`:

```python
from cosmos_tokenizer.networks.configs import (
    COSMOS_CI_8x8,     # Continuous Image 8×8
    COSMOS_CI_16x16,   # Continuous Image 16×16
    COSMOS_DI_8x8,     # Discrete Image 8×8
    COSMOS_DI_16x16,   # Discrete Image 16×16
    COSMOS_CV_4x8x8,   # Continuous Video 4×8×8
    COSMOS_CV_8x8x8,   # Continuous Video 8×8×8
    COSMOS_DV_8x16x16, # Discrete Video 8×16×16
)
```

### Fields

```python
@dataclass
class TokenizerConfig:
    mode: str                       # "CI", "DI", "CV", "DV"
    spatial_compression: int        # 8 or 16
    temporal_compression: int = 1   # 1 (image), 4, or 8 (video)

    # Architecture
    ch: int = 128                   # Base channel count
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2         # ResNet blocks per stage
    attn_resolutions: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.0

    # Latent space
    z_channels: int = 16            # Latent channels (CI/CV)
    double_z: bool = True           # For CI/CV: output mean+logvar

    # Quantization (DI/DV only)
    quantizer_type: str = "fsq"     # "vq", "fsq", "lfq", "residual_fsq"
    fsq_levels: List[int] = field(default_factory=lambda: [8,8,8,5,5,5])
    codebook_size: int = 64000      # For VQ

    # Patching
    patch_size: int = 4             # Haar wavelet patch size
    patch_method: str = "haar"      # "haar" or "rearrange"

    # Video-specific
    causal: bool = False            # Use causal convolutions
```

### Custom Config Example

```python
from cosmos_tokenizer import ImageTokenizer, TokenizerConfig

# Create custom config
custom_config = TokenizerConfig(
    mode="DI",
    spatial_compression=16,
    z_channels=8,  # Smaller latent space
    ch=64,         # Fewer channels (lighter model)
    ch_mult=[1, 2, 4],
    num_res_blocks=3,  # More residual blocks
    quantizer_type="vq",  # Use VQ instead of FSQ
    codebook_size=8192,   # Smaller codebook
)

tokenizer = ImageTokenizer(config=custom_config)
```

---

## Quantizers

Low-level quantization modules (advanced users).

### VectorQuantizer

Classical VQ-VAE with EMA codebook updates.

```python
from cosmos_tokenizer.modules.quantizers import VectorQuantizer

quantizer = VectorQuantizer(
    codebook_size=64000,
    embedding_dim=6,
    commitment_cost=0.25,
    use_ema=True,
    decay=0.99,
)

# Forward
z = torch.randn(4, 6, 16, 16)  # (B, D, H, W)
z_q, indices, info = quantizer(z)
# z_q: quantized, indices: codebook indices, info: perplexity/loss
```

### FSQuantizer

Finite Scalar Quantization (implicit codebook).

```python
from cosmos_tokenizer.modules.quantizers import FSQuantizer

quantizer = FSQuantizer(
    levels=[8, 8, 8, 5, 5, 5],  # 8*8*8*5*5*5 = 64K
    dim=6,
)

z = torch.randn(4, 6, 16, 16)
z_q, indices, info = quantizer(z)
```

### LFQuantizer

Lookup-Free binary quantization.

```python
from cosmos_tokenizer.modules.quantizers import LFQuantizer

quantizer = LFQuantizer(
    dim=16,  # 2^16 = 65536 tokens
    entropy_loss_weight=0.1,
)

z = torch.randn(4, 16, 16, 16)
z_q, indices, info = quantizer(z)
```

### ResidualFSQuantizer

Multi-stage residual FSQ.

```python
from cosmos_tokenizer.modules.quantizers import ResidualFSQuantizer

quantizer = ResidualFSQuantizer(
    dim=6,
    levels=[8, 5, 5],  # Per stage
    num_stages=4,
)

z = torch.randn(4, 6, 16, 16)
z_q, indices_list, info = quantizer(z)
# indices_list: list of indices per stage
```

---

## Network Classes

Low-level model classes (advanced usage).

### ContinuousImageTokenizer

```python
from cosmos_tokenizer.networks.autoencoder import ContinuousImageTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_CI_16x16

model = ContinuousImageTokenizer(COSMOS_CI_16x16)
reconstruction, info = model(images)
```

### DiscreteImageTokenizer

```python
from cosmos_tokenizer.networks.autoencoder import DiscreteImageTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_DI_16x16

model = DiscreteImageTokenizer(COSMOS_DI_16x16)
reconstruction, info = model(images)
```

### ContinuousVideoTokenizer

```python
from cosmos_tokenizer.networks.autoencoder import ContinuousVideoTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_CV_8x8x8

model = ContinuousVideoTokenizer(COSMOS_CV_8x8x8)
reconstruction, info = model(videos)
```

### DiscreteVideoTokenizer

```python
from cosmos_tokenizer.networks.autoencoder import DiscreteVideoTokenizer
from cosmos_tokenizer.networks.configs import COSMOS_DV_8x16x16

model = DiscreteVideoTokenizer(COSMOS_DV_8x16x16)
reconstruction, info = model(videos)
```

---

## Loss Functions

Training losses (for model training).

### TokenizerLoss

Combined loss function for training.

```python
from cosmos_tokenizer.training.losses import TokenizerLoss

criterion = TokenizerLoss(
    recon_loss_type="l1",         # "l1", "l2", "smooth_l1"
    perceptual_weight=1.0,        # Weight for perceptual loss
    quantizer_weight=1.0,         # Weight for quantizer loss
    kl_weight=1e-6,               # Weight for KL divergence (CI/CV only)
    use_perceptual=True,          # Enable VGG perceptual loss
)

# Training step
reconstruction, info = model(images)
losses = criterion(reconstruction, images, info)

total_loss = losses["total_loss"]
recon_loss = losses["recon_loss"]
perceptual_loss = losses["perceptual_loss"]
quantizer_loss = losses["quantizer_loss"]  # DI/DV only
kl_loss = losses["kl_loss"]                # CI/CV only
```

---

## Data Types and Shapes

### Image Tensors

**Input images:**
- Shape: `(B, 3, H, W)`
- Range: `[-1, 1]` (NOT `[0, 1]` or `[0, 255]`)
- Channels: RGB order
- Height/Width: Should be divisible by `spatial_compression * patch_size` (typically multiples of 64)
- Recommended sizes: 256, 512, 1024

**Convert from PIL:**
```python
from PIL import Image
import numpy as np
import torch

img = Image.open("photo.jpg").convert("RGB")
img = img.resize((256, 256))
img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]
img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
img_tensor = img_tensor * 2.0 - 1.0  # [-1, 1]
img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
```

**Convert back to PIL:**
```python
def tensor_to_pil(tensor):
    # tensor: (B, 3, H, W) or (3, H, W), range [-1, 1]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = (tensor + 1.0) / 2.0  # -> [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)
```

### Video Tensors

**Input videos:**
- Shape: `(B, 3, T, H, W)`
- Range: `[-1, 1]`
- `T`: Number of frames
- Recommended `T`: 17, 33 (odd numbers work better with causal processing)

**Load from file (OpenCV example):**
```python
import cv2

cap = cv2.VideoCapture("video.mp4")
frames = []
while len(frames) < 32:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frames.append(frame)
cap.release()

video_array = np.array(frames).astype(np.float32) / 255.0
video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)  # (T, 3, H, W)
video_tensor = video_tensor.permute(1, 0, 2, 3)  # (3, T, H, W)
video_tensor = video_tensor * 2.0 - 1.0
video_tensor = video_tensor.unsqueeze(0)  # (1, 3, T, H, W)
```

---

## Common Patterns

### Batch Processing

```python
from torch.utils.data import DataLoader

tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, device="cuda")

# Your dataset
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

all_tokens = []
for batch in dataloader:
    images = batch["image"].cuda()  # (32, 3, 256, 256)
    tokens = tokenizer.encode(images)  # (32, 16, 16)
    all_tokens.append(tokens.cpu())

all_tokens = torch.cat(all_tokens, dim=0)
```

### Mixed Precision

```python
from torch.cuda.amp import autocast

tokenizer = ImageTokenizer(mode="DI", spatial_compression=16, dtype="bfloat16")

with autocast():
    tokens = tokenizer.encode(images)
    reconstructed = tokenizer.decode(tokens)
```

### Distributed Training

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DiscreteImageTokenizer(config)
model = DDP(model, device_ids=[local_rank])

# Training loop
for batch in dataloader:
    reconstruction, info = model(images)
    loss = criterion(reconstruction, images, info)["total_loss"]
    loss.backward()
    optimizer.step()
```

---

## Version Compatibility

- **PyTorch:** >= 2.0.0
- **Python:** >= 3.10
- **CUDA:** >= 11.8 (for GPU support)
- **einops:** >= 0.7.0
- **einx:** >= 0.1.3

---

## See Also

- [Getting Started Guide](GETTING_STARTED.md) - Beginner tutorial
- [Examples](EXAMPLES.md) - Practical code examples
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [FAQ](FAQ.md) - Frequently asked questions

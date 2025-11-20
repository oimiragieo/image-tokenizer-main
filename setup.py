"""Setup script for Cosmos Image Tokenizer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "einops>=0.7.0",
    "einx>=0.1.3",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "loguru>=0.7.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "huggingface-hub>=0.20.0",
    "safetensors>=0.4.0",
]

# Video processing extras
video_requires = [
    "av>=10.0.0",
    "opencv-python>=4.8.0",
    "mediapy>=1.1.6",
    "decord>=0.6.0",
]

# Training extras
train_requires = [
    "pytorch-lightning>=2.0.0",
    "wandb>=0.15.0",
    "tensorboard>=2.14.0",
    "torchmetrics>=1.0.0",
    "lpips>=0.1.4",
    "timm>=0.9.0",
]

# Object segmentation extras
segment_requires = [
    "segment-anything>=1.0",
    "transformers>=4.35.0",
]

# Development extras
dev_requires = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    "ipython>=8.14.0",
    "jupyter>=1.0.0",
]

# All extras
all_requires = (
    install_requires
    + video_requires
    + train_requires
    + segment_requires
    + dev_requires
)

setup(
    name="cosmos-tokenizer",
    version="1.0.0",
    author="Cosmos Tokenizer Team",
    description="World-class image and video tokenizer with state-of-the-art compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cosmos-image-tokenizer",
    packages=find_packages(exclude=["tests", "scripts", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "video": video_requires,
        "train": train_requires,
        "segment": segment_requires,
        "dev": dev_requires,
        "all": all_requires,
    },
    entry_points={
        "console_scripts": [
            "cosmos-image=cosmos_tokenizer.cli.image_cli:main",
            "cosmos-video=cosmos_tokenizer.cli.video_cli:main",
            "cosmos-train=cosmos_tokenizer.cli.train_cli:main",
            "cosmos-benchmark=cosmos_tokenizer.cli.benchmark_cli:main",
        ],
    },
)

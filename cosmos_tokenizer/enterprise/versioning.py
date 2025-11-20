"""
Versioning and metadata management for tokenizer models.

Provides:
- Model versioning system
- Metadata tracking (codebook size, compression, etc.)
- Backward compatibility checks
- Version migration utilities
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class TokenizerMetadata:
    """
    Comprehensive metadata for tokenizer models.

    Attributes:
        version: Model version (semver format: major.minor.patch)
        modality: Modality type (image, video, audio, text)
        mode: Tokenizer mode (CI, DI, CV, DV)
        spatial_compression: Spatial compression factor
        temporal_compression: Temporal compression factor (video only)
        codebook_size: Size of quantization codebook
        patch_size: Patch size for preprocessing
        z_channels: Number of latent channels
        model_name: Human-readable model name
        description: Model description
        created_at: Creation timestamp
        updated_at: Last update timestamp
        checkpoint_path: Path to model weights
        config_path: Path to configuration file
        backward_compatible_with: List of compatible versions
        tags: Custom tags for organization
    """

    version: str
    modality: str
    mode: str
    spatial_compression: int
    temporal_compression: Optional[int] = None
    codebook_size: Optional[int] = None
    patch_size: int = 1
    z_channels: int = 16
    model_name: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    checkpoint_path: str = ""
    config_path: str = ""
    backward_compatible_with: list = None
    tags: list = None

    def __post_init__(self):
        if self.backward_compatible_with is None:
            self.backward_compatible_with = []
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Path):
        """Save metadata to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "TokenizerMetadata":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> "TokenizerMetadata":
        """Load metadata from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now().isoformat()

    def is_compatible_with(self, other_version: str) -> bool:
        """
        Check if this version is compatible with another.

        Args:
            other_version: Version to check compatibility with

        Returns:
            True if compatible
        """
        return other_version in self.backward_compatible_with or other_version == self.version

    def get_compression_info(self) -> Dict:
        """Get compression configuration."""
        info = {
            "spatial_compression": self.spatial_compression,
            "patch_size": self.patch_size,
            "total_spatial_compression": self.spatial_compression * self.patch_size,
        }

        if self.temporal_compression:
            info["temporal_compression"] = self.temporal_compression
            info["total_compression"] = (
                self.spatial_compression ** 2
            ) * self.temporal_compression

        return info


class VersionManager:
    """
    Manage tokenizer versions and metadata.

    Features:
    - Version registry
    - Compatibility checks
    - Metadata storage
    - Version migration

    Usage:
        manager = VersionManager(registry_path="models/registry.yaml")

        # Register a model
        metadata = TokenizerMetadata(
            version="1.0.0",
            modality="image",
            mode="DI",
            spatial_compression=16,
            codebook_size=65536,
            model_name="Cosmos-DI-16x16",
        )
        manager.register(metadata)

        # Get model by version
        model_meta = manager.get("1.0.0")

        # Check compatibility
        is_compatible = manager.check_compatibility("1.0.0", "1.1.0")
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize version manager.

        Args:
            registry_path: Path to version registry file
        """
        self.registry_path = (
            Path(registry_path) if registry_path else Path("metadata/registry.yaml")
        )
        self._registry: Dict[str, TokenizerMetadata] = {}
        self._load_registry()

    def _load_registry(self):
        """Load version registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = yaml.safe_load(f)
                if data:
                    for version, meta_dict in data.items():
                        self._registry[version] = TokenizerMetadata.from_dict(
                            meta_dict
                        )

    def _save_registry(self):
        """Save version registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_dict = {
            version: meta.to_dict() for version, meta in self._registry.items()
        }
        with open(self.registry_path, "w") as f:
            yaml.dump(registry_dict, f, default_flow_style=False)

    def register(
        self,
        metadata: TokenizerMetadata,
        overwrite: bool = False,
    ):
        """
        Register a tokenizer version.

        Args:
            metadata: Tokenizer metadata
            overwrite: Allow overwriting existing version

        Raises:
            ValueError: If version already exists and overwrite=False
        """
        if metadata.version in self._registry and not overwrite:
            raise ValueError(
                f"Version {metadata.version} already registered. "
                f"Use overwrite=True to replace."
            )

        self._registry[metadata.version] = metadata
        self._save_registry()

    def get(self, version: str) -> TokenizerMetadata:
        """
        Get metadata for a specific version.

        Args:
            version: Version string

        Returns:
            TokenizerMetadata

        Raises:
            KeyError: If version not found
        """
        if version not in self._registry:
            raise KeyError(
                f"Version {version} not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[version]

    def list_versions(
        self,
        modality: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> list[str]:
        """
        List registered versions.

        Args:
            modality: Filter by modality
            mode: Filter by mode

        Returns:
            List of version strings
        """
        versions = []

        for version, meta in self._registry.items():
            if modality and meta.modality != modality:
                continue
            if mode and meta.mode != mode:
                continue
            versions.append(version)

        return sorted(versions)

    def check_compatibility(
        self,
        version1: str,
        version2: str,
    ) -> bool:
        """
        Check if two versions are compatible.

        Args:
            version1: First version
            version2: Second version

        Returns:
            True if compatible
        """
        meta1 = self.get(version1)
        return meta1.is_compatible_with(version2)

    def get_latest(
        self,
        modality: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Optional[TokenizerMetadata]:
        """
        Get latest version metadata.

        Args:
            modality: Filter by modality
            mode: Filter by mode

        Returns:
            TokenizerMetadata for latest version, or None if no versions found
        """
        versions = self.list_versions(modality, mode)

        if not versions:
            return None

        # Simple lexicographic sort (for semver: "1.10.0" > "1.2.0")
        latest_version = max(
            versions,
            key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit()),
        )

        return self.get(latest_version)

    def export_metadata(
        self,
        version: str,
        output_path: Path,
    ):
        """
        Export metadata for a specific version.

        Args:
            version: Version to export
            output_path: Output file path
        """
        metadata = self.get(version)
        metadata.to_yaml(output_path)

    def import_metadata(
        self,
        input_path: Path,
        overwrite: bool = False,
    ):
        """
        Import metadata from file.

        Args:
            input_path: Input file path
            overwrite: Allow overwriting existing version
        """
        metadata = TokenizerMetadata.from_yaml(input_path)
        self.register(metadata, overwrite=overwrite)

"""
Safety validation layer for multimodal tokenization.

Provides:
- Input validation (size, resolution, format)
- Safety checks (NSFW detection hooks, malicious input detection)
- Rate limiting
- Audit logging
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import torch

from cosmos_tokenizer.core.modality import Modality


@dataclass
class ValidationResult:
    """
    Result from safety validation.

    Attributes:
        is_valid: Whether input passed validation
        error_message: Error message if validation failed
        warnings: List of warning messages
        metadata: Additional validation metadata
    """

    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class SafetyValidator:
    """
    Safety validation for multimodal inputs.

    Features:
    - File size and resolution limits
    - Input sanitization
    - Rate limiting
    - NSFW detection hooks
    - Audit logging

    Usage:
        validator = SafetyValidator(
            max_image_resolution=4096,
            max_video_resolution=2048,
            max_file_size_mb=100.0,
        )

        # Validate image
        result = validator.validate_image(image)
        if not result.is_valid:
            print(f"Validation failed: {result.error_message}")

        # Check rate limit
        can_proceed, wait_time = validator.check_rate_limit(user_id="user-123")
        if not can_proceed:
            print(f"Rate limited. Try again in {wait_time}s")
    """

    def __init__(
        self,
        max_image_resolution: int = 4096,
        max_video_resolution: int = 2048,
        max_video_frames: int = 1000,
        max_file_size_mb: float = 100.0,
        enable_nsfw_detection: bool = False,
        rate_limit_requests_per_minute: int = 60,
    ):
        """
        Initialize safety validator.

        Args:
            max_image_resolution: Maximum image size (width or height)
            max_video_resolution: Maximum video resolution
            max_video_frames: Maximum number of frames
            max_file_size_mb: Maximum file size in MB
            enable_nsfw_detection: Enable NSFW detection (requires external service)
            rate_limit_requests_per_minute: Rate limit per user
        """
        self.max_image_resolution = max_image_resolution
        self.max_video_resolution = max_video_resolution
        self.max_video_frames = max_video_frames
        self.max_file_size_mb = max_file_size_mb
        self.enable_nsfw_detection = enable_nsfw_detection
        self.rate_limit = rate_limit_requests_per_minute

        # Rate limiting tracking
        self._user_requests: Dict[str, List[datetime]] = {}

        # Audit log
        self._audit_log: List[Dict] = []

    def validate_image(
        self,
        image: torch.Tensor,
        user_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate image input.

        Args:
            image: Image tensor (B, 3, H, W)
            user_id: User identifier for audit log

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # Check tensor type
        if not isinstance(image, torch.Tensor):
            return ValidationResult(
                is_valid=False,
                error_message="Input must be a torch.Tensor",
            )

        # Check dimensions
        if image.ndim != 4:
            return ValidationResult(
                is_valid=False,
                error_message=f"Expected 4D tensor (B,C,H,W), got {image.ndim}D",
            )

        batch, channels, height, width = image.shape

        # Check channels
        if channels != 3:
            return ValidationResult(
                is_valid=False,
                error_message=f"Expected 3 channels (RGB), got {channels}",
            )

        # Check resolution
        if max(height, width) > self.max_image_resolution:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Resolution {height}x{width} exceeds "
                    f"maximum {self.max_image_resolution}"
                ),
            )

        # Check file size (estimate)
        size_mb = (image.numel() * image.element_size()) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Estimated size {size_mb:.1f}MB exceeds "
                    f"maximum {self.max_file_size_mb}MB"
                ),
            )

        # Check value range
        min_val, max_val = image.min().item(), image.max().item()
        if min_val < -1.1 or max_val > 1.1:
            warnings.append(
                f"Values outside expected range [-1, 1]: [{min_val:.2f}, {max_val:.2f}]"
            )

        # Check for NaN/Inf
        if torch.isnan(image).any():
            return ValidationResult(
                is_valid=False,
                error_message="Input contains NaN values",
            )

        if torch.isinf(image).any():
            return ValidationResult(
                is_valid=False,
                error_message="Input contains Inf values",
            )

        # NSFW detection hook (placeholder)
        if self.enable_nsfw_detection:
            is_safe, nsfw_score = self._check_nsfw(image)
            if not is_safe:
                self._log_audit(
                    modality=Modality.IMAGE,
                    user_id=user_id,
                    action="nsfw_blocked",
                    metadata={"nsfw_score": nsfw_score},
                )
                return ValidationResult(
                    is_valid=False,
                    error_message="Content violates safety policies",
                    metadata={"nsfw_score": nsfw_score},
                )

        # Log successful validation
        self._log_audit(
            modality=Modality.IMAGE,
            user_id=user_id,
            action="validated",
            metadata={"shape": tuple(image.shape)},
        )

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            metadata={"shape": tuple(image.shape), "size_mb": size_mb},
        )

    def validate_video(
        self,
        video: torch.Tensor,
        user_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate video input.

        Args:
            video: Video tensor (B, 3, T, H, W)
            user_id: User identifier for audit log

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # Check tensor type
        if not isinstance(video, torch.Tensor):
            return ValidationResult(
                is_valid=False,
                error_message="Input must be a torch.Tensor",
            )

        # Check dimensions
        if video.ndim != 5:
            return ValidationResult(
                is_valid=False,
                error_message=f"Expected 5D tensor (B,C,T,H,W), got {video.ndim}D",
            )

        batch, channels, frames, height, width = video.shape

        # Check channels
        if channels != 3:
            return ValidationResult(
                is_valid=False,
                error_message=f"Expected 3 channels (RGB), got {channels}",
            )

        # Check resolution
        if max(height, width) > self.max_video_resolution:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Resolution {height}x{width} exceeds "
                    f"maximum {self.max_video_resolution}"
                ),
            )

        # Check frame count
        if frames > self.max_video_frames:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Frame count {frames} exceeds maximum {self.max_video_frames}"
                ),
            )

        # Check file size (estimate)
        size_mb = (video.numel() * video.element_size()) / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Estimated size {size_mb:.1f}MB exceeds "
                    f"maximum {self.max_file_size_mb}MB"
                ),
            )

        # Check value range
        min_val, max_val = video.min().item(), video.max().item()
        if min_val < -1.1 or max_val > 1.1:
            warnings.append(
                f"Values outside expected range [-1, 1]: [{min_val:.2f}, {max_val:.2f}]"
            )

        # Check for NaN/Inf
        if torch.isnan(video).any():
            return ValidationResult(
                is_valid=False,
                error_message="Input contains NaN values",
            )

        if torch.isinf(video).any():
            return ValidationResult(
                is_valid=False,
                error_message="Input contains Inf values",
            )

        # Log successful validation
        self._log_audit(
            modality=Modality.VIDEO,
            user_id=user_id,
            action="validated",
            metadata={"shape": tuple(video.shape)},
        )

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            metadata={"shape": tuple(video.shape), "size_mb": size_mb},
        )

    def check_rate_limit(
        self,
        user_id: str,
    ) -> Tuple[bool, float]:
        """
        Check if user is within rate limit.

        Args:
            user_id: User identifier

        Returns:
            (can_proceed, wait_seconds) tuple
        """
        now = datetime.now()
        window = timedelta(minutes=1)

        # Clean old requests
        if user_id in self._user_requests:
            self._user_requests[user_id] = [
                req for req in self._user_requests[user_id] if now - req < window
            ]
        else:
            self._user_requests[user_id] = []

        # Check limit
        request_count = len(self._user_requests[user_id])

        if request_count >= self.rate_limit:
            # Calculate wait time
            oldest_request = min(self._user_requests[user_id])
            wait_time = (oldest_request + window - now).total_seconds()
            return False, max(0, wait_time)

        # Record request
        self._user_requests[user_id].append(now)
        return True, 0.0

    def _check_nsfw(
        self,
        image: torch.Tensor,
    ) -> Tuple[bool, float]:
        """
        NSFW detection hook (placeholder for external service).

        Args:
            image: Image to check

        Returns:
            (is_safe, nsfw_score) tuple

        Note:
            In production, this would integrate with services like:
            - Azure Content Moderator
            - AWS Rekognition
            - Google Cloud Vision Safe Search
            - CLIP-based NSFW classifier
        """
        # Placeholder - always returns safe
        # In production, integrate with external NSFW detection service
        return True, 0.0

    def _log_audit(
        self,
        modality: Modality,
        user_id: Optional[str],
        action: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Log audit event.

        Args:
            modality: Modality type
            user_id: User identifier
            action: Action performed
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "modality": modality.value,
            "user_id": user_id or "anonymous",
            "action": action,
            "metadata": metadata or {},
        }

        self._audit_log.append(log_entry)

    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        modality: Optional[Modality] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Get filtered audit log.

        Args:
            user_id: Filter by user
            modality: Filter by modality
            since: Filter by time

        Returns:
            List of audit log entries
        """
        logs = self._audit_log

        if user_id is not None:
            logs = [log for log in logs if log["user_id"] == user_id]

        if modality is not None:
            logs = [log for log in logs if log["modality"] == modality.value]

        if since is not None:
            since_iso = since.isoformat()
            logs = [log for log in logs if log["timestamp"] >= since_iso]

        return logs

    def reset_audit_log(self):
        """Clear audit log."""
        self._audit_log.clear()

    def reset_rate_limits(self):
        """Clear rate limit tracking."""
        self._user_requests.clear()

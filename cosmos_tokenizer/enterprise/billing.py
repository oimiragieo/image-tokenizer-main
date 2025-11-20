"""
Billing engine for multimodal tokenization services.

Provides:
- Cost calculation based on token usage
- Pricing tiers (free, pro, enterprise)
- Quota management
- Billing reports
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from cosmos_tokenizer.core.modality import Modality


class PricingTier(str, Enum):
    """Pricing tiers for token usage."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class PricingConfig:
    """
    Pricing configuration for token usage.

    Prices are in USD per million tokens.
    """

    # Image tokens (per million)
    image_encode_price: float = 0.10  # $0.10 per 1M tokens
    image_decode_price: float = 0.05  # $0.05 per 1M tokens

    # Video tokens (per million)
    video_encode_price: float = 1.00  # $1.00 per 1M tokens
    video_decode_price: float = 0.50  # $0.50 per 1M tokens

    # Audio tokens (per million)
    audio_encode_price: float = 0.20  # $0.20 per 1M tokens
    audio_decode_price: float = 0.10  # $0.10 per 1M tokens

    # Text tokens (per million)
    text_encode_price: float = 0.01  # $0.01 per 1M tokens
    text_decode_price: float = 0.01  # $0.01 per 1M tokens

    # Quotas (tokens per month)
    free_quota: int = 1_000_000  # 1M tokens/month
    pro_quota: int = 100_000_000  # 100M tokens/month
    enterprise_quota: int = -1  # Unlimited

    @classmethod
    def get_tier_config(cls, tier: PricingTier) -> "PricingConfig":
        """Get pricing configuration for a tier."""
        if tier == PricingTier.FREE:
            return cls(
                image_encode_price=0.0,  # Free
                image_decode_price=0.0,
                video_encode_price=0.0,
                video_decode_price=0.0,
                audio_encode_price=0.0,
                audio_decode_price=0.0,
                text_encode_price=0.0,
                text_decode_price=0.0,
            )
        elif tier == PricingTier.PRO:
            return cls()  # Default prices
        else:  # ENTERPRISE
            # Enterprise gets discounted rates (30% off)
            return cls(
                image_encode_price=0.07,
                image_decode_price=0.035,
                video_encode_price=0.70,
                video_decode_price=0.35,
                audio_encode_price=0.14,
                audio_decode_price=0.07,
                text_encode_price=0.007,
                text_decode_price=0.007,
            )


class CostCalculator:
    """
    Calculate costs based on token usage.

    Usage:
        calculator = CostCalculator(tier=PricingTier.PRO)

        # Calculate cost for image encoding
        cost = calculator.calculate_cost(
            modality=Modality.IMAGE,
            token_count=10_000,
            operation="encode",
        )

        print(f"Cost: ${cost:.4f}")
    """

    def __init__(self, tier: PricingTier = PricingTier.PRO):
        """
        Initialize cost calculator.

        Args:
            tier: Pricing tier to use
        """
        self.tier = tier
        self.config = PricingConfig.get_tier_config(tier)

    def calculate_cost(
        self,
        modality: Modality,
        token_count: int,
        operation: str,
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            modality: Modality type
            token_count: Number of tokens
            operation: Operation type ('encode' or 'decode')

        Returns:
            Cost in USD
        """
        # Get price per million tokens
        price_key = f"{modality.value}_{operation}_price"
        price_per_million = getattr(self.config, price_key, 0.0)

        # Calculate cost
        cost = (token_count / 1_000_000) * price_per_million

        return cost

    def calculate_batch_cost(
        self,
        usage_records: list[Dict],
    ) -> Dict[str, float]:
        """
        Calculate costs for multiple usage records.

        Args:
            usage_records: List of usage records with
                          (modality, token_count, operation)

        Returns:
            Dictionary with cost breakdown
        """
        total_cost = 0.0
        cost_by_modality: Dict[str, float] = {}
        cost_by_operation: Dict[str, float] = {}

        for record in usage_records:
            modality = record["modality"]
            if isinstance(modality, str):
                modality = Modality(modality)

            token_count = record["token_count"]
            operation = record["operation"]

            cost = self.calculate_cost(modality, token_count, operation)
            total_cost += cost

            # Aggregate by modality
            modality_key = modality.value
            cost_by_modality[modality_key] = (
                cost_by_modality.get(modality_key, 0.0) + cost
            )

            # Aggregate by operation
            cost_by_operation[operation] = cost_by_operation.get(operation, 0.0) + cost

        return {
            "total_cost": total_cost,
            "by_modality": cost_by_modality,
            "by_operation": cost_by_operation,
        }

    def check_quota(
        self,
        current_usage: int,
    ) -> tuple[bool, int, Optional[str]]:
        """
        Check if usage is within quota.

        Args:
            current_usage: Current token usage this month

        Returns:
            (within_quota, remaining_tokens, error_message) tuple
        """
        if self.tier == PricingTier.FREE:
            quota = self.config.free_quota
        elif self.tier == PricingTier.PRO:
            quota = self.config.pro_quota
        else:  # ENTERPRISE
            quota = self.config.enterprise_quota

        # Unlimited quota
        if quota < 0:
            return True, -1, None

        # Check quota
        if current_usage > quota:
            return (
                False,
                0,
                f"Quota exceeded: {current_usage:,} / {quota:,} tokens used",
            )

        remaining = quota - current_usage
        return True, remaining, None


class BillingEngine:
    """
    Enterprise billing engine for multimodal tokenization.

    Features:
    - Cost tracking across modalities
    - Quota enforcement
    - Billing reports
    - Usage analytics

    Usage:
        engine = BillingEngine(tier=PricingTier.PRO)

        # Record usage
        engine.record_usage(
            modality=Modality.IMAGE,
            token_count=10_000,
            operation="encode",
            user_id="user-123",
        )

        # Get bill
        bill = engine.generate_bill(user_id="user-123")
        print(f"Total cost: ${bill['total_cost']:.2f}")

        # Check quota
        can_proceed, remaining, msg = engine.check_quota("user-123")
    """

    def __init__(self, tier: PricingTier = PricingTier.PRO):
        """
        Initialize billing engine.

        Args:
            tier: Pricing tier
        """
        self.tier = tier
        self.calculator = CostCalculator(tier)
        self._usage_by_user: Dict[str, list[Dict]] = {}
        self._total_usage_by_user: Dict[str, int] = {}

    def record_usage(
        self,
        modality: Modality,
        token_count: int,
        operation: str,
        user_id: str = "default",
        metadata: Optional[Dict] = None,
    ):
        """
        Record token usage for billing.

        Args:
            modality: Modality type
            token_count: Number of tokens
            operation: Operation type
            user_id: User identifier
            metadata: Additional metadata
        """
        record = {
            "modality": modality,
            "token_count": token_count,
            "operation": operation,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }

        # Add to user's records
        if user_id not in self._usage_by_user:
            self._usage_by_user[user_id] = []
            self._total_usage_by_user[user_id] = 0

        self._usage_by_user[user_id].append(record)
        self._total_usage_by_user[user_id] += token_count

    def check_quota(
        self,
        user_id: str = "default",
    ) -> tuple[bool, int, Optional[str]]:
        """
        Check if user is within quota.

        Args:
            user_id: User identifier

        Returns:
            (within_quota, remaining_tokens, error_message) tuple
        """
        current_usage = self._total_usage_by_user.get(user_id, 0)
        return self.calculator.check_quota(current_usage)

    def generate_bill(
        self,
        user_id: str = "default",
    ) -> Dict:
        """
        Generate billing report for a user.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with billing information
        """
        if user_id not in self._usage_by_user:
            return {
                "user_id": user_id,
                "tier": self.tier.value,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_modality": {},
                "by_operation": {},
                "records": [],
            }

        records = self._usage_by_user[user_id]
        cost_breakdown = self.calculator.calculate_batch_cost(records)

        return {
            "user_id": user_id,
            "tier": self.tier.value,
            "total_tokens": self._total_usage_by_user[user_id],
            "total_cost": cost_breakdown["total_cost"],
            "by_modality": cost_breakdown["by_modality"],
            "by_operation": cost_breakdown["by_operation"],
            "records": records,
        }

    def get_all_bills(self) -> Dict[str, Dict]:
        """Generate bills for all users."""
        return {user_id: self.generate_bill(user_id) for user_id in self._usage_by_user}

    def reset_user(self, user_id: str):
        """Reset usage for a specific user."""
        if user_id in self._usage_by_user:
            del self._usage_by_user[user_id]
            del self._total_usage_by_user[user_id]

    def reset_all(self):
        """Reset all usage records."""
        self._usage_by_user.clear()
        self._total_usage_by_user.clear()

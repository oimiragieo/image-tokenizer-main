"""
Usage tracking and analytics for multimodal tokenization.

Provides:
- Real-time usage monitoring
- Performance metrics collection
- Analytics aggregation
- Export to monitoring systems
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from cosmos_tokenizer.core.modality import Modality


@dataclass
class UsageRecord:
    """
    Detailed usage record for analytics.

    Attributes:
        modality: Modality type
        operation: Operation type ('encode' or 'decode')
        token_count: Number of tokens
        latency_ms: Operation latency in milliseconds
        throughput_tokens_per_sec: Tokens processed per second
        memory_mb: Peak memory usage in MB
        device: Device used (cuda, cpu, mps)
        user_id: User identifier
        timestamp: When the operation occurred
        metadata: Additional information
    """

    modality: Modality
    operation: str
    token_count: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_mb: Optional[float] = None
    device: str = "cuda"
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "modality": self.modality.value,
            "operation": self.operation,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_mb": self.memory_mb,
            "device": self.device,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class UsageTracker:
    """
    Track usage and performance metrics for multimodal tokenization.

    Features:
    - Real-time metrics collection
    - Performance analytics
    - Usage aggregation
    - Time-series analysis

    Usage:
        tracker = UsageTracker()

        # Record usage
        tracker.record(
            modality=Modality.IMAGE,
            operation="encode",
            token_count=256,
            latency_ms=15.5,
            throughput_tokens_per_sec=16516.0,
            user_id="user-123",
        )

        # Get metrics
        metrics = tracker.get_metrics()
        print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"Total tokens: {metrics['total_tokens']:,}")
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize usage tracker.

        Args:
            retention_hours: How long to keep records (default 24 hours)
        """
        self._records: List[UsageRecord] = []
        self.retention_hours = retention_hours

    def record(
        self,
        modality: Modality,
        operation: str,
        token_count: int,
        latency_ms: float,
        throughput_tokens_per_sec: float,
        memory_mb: Optional[float] = None,
        device: str = "cuda",
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            modality: Modality type
            operation: Operation type
            token_count: Number of tokens
            latency_ms: Operation latency
            throughput_tokens_per_sec: Throughput
            memory_mb: Memory usage
            device: Device used
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            UsageRecord instance
        """
        record = UsageRecord(
            modality=modality,
            operation=operation,
            token_count=token_count,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            memory_mb=memory_mb,
            device=device,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._cleanup_old_records()

        return record

    def _cleanup_old_records(self):
        """Remove records older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        self._records = [r for r in self._records if r.timestamp >= cutoff]

    def get_records(
        self,
        modality: Optional[Modality] = None,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[UsageRecord]:
        """
        Get filtered usage records.

        Args:
            modality: Filter by modality
            operation: Filter by operation
            user_id: Filter by user
            since: Filter by time (records after this time)

        Returns:
            List of matching records
        """
        records = self._records

        if modality is not None:
            records = [r for r in records if r.modality == modality]

        if operation is not None:
            records = [r for r in records if r.operation == operation]

        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]

        if since is not None:
            records = [r for r in records if r.timestamp >= since]

        return records

    def get_metrics(
        self,
        modality: Optional[Modality] = None,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> Dict:
        """
        Calculate aggregate metrics.

        Args:
            modality: Filter by modality
            operation: Filter by operation
            user_id: Filter by user
            since: Filter by time

        Returns:
            Dictionary with aggregate metrics
        """
        records = self.get_records(modality, operation, user_id, since)

        if not records:
            return {
                "count": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0.0,
                "avg_throughput": 0.0,
                "avg_memory_mb": 0.0,
            }

        total_tokens = sum(r.token_count for r in records)
        latencies = [r.latency_ms for r in records]
        throughputs = [r.throughput_tokens_per_sec for r in records]
        memories = [r.memory_mb for r in records if r.memory_mb is not None]

        return {
            "count": len(records),
            "total_tokens": total_tokens,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "min_throughput": min(throughputs),
            "max_throughput": max(throughputs),
            "avg_memory_mb": sum(memories) / len(memories) if memories else 0.0,
        }

    def get_metrics_by_modality(self) -> Dict[str, Dict]:
        """Get metrics broken down by modality."""
        return {
            modality.value: self.get_metrics(modality=modality)
            for modality in Modality
            if self.get_records(modality=modality)
        }

    def get_metrics_by_operation(self) -> Dict[str, Dict]:
        """Get metrics broken down by operation."""
        operations = set(r.operation for r in self._records)
        return {
            operation: self.get_metrics(operation=operation) for operation in operations
        }

    def get_metrics_by_user(self) -> Dict[str, Dict]:
        """Get metrics broken down by user."""
        users = set(r.user_id for r in self._records if r.user_id)
        return {user: self.get_metrics(user_id=user) for user in users}

    def get_time_series(
        self,
        interval_minutes: int = 5,
        modality: Optional[Modality] = None,
    ) -> List[Dict]:
        """
        Get time-series data for visualization.

        Args:
            interval_minutes: Time bucket size in minutes
            modality: Filter by modality

        Returns:
            List of time buckets with aggregated metrics
        """
        records = self.get_records(modality=modality)

        if not records:
            return []

        # Find time range
        min_time = min(r.timestamp for r in records)
        max_time = max(r.timestamp for r in records)

        # Create time buckets
        interval = timedelta(minutes=interval_minutes)
        buckets = []
        current_time = min_time

        while current_time <= max_time:
            bucket_end = current_time + interval
            bucket_records = [
                r for r in records if current_time <= r.timestamp < bucket_end
            ]

            if bucket_records:
                total_tokens = sum(r.token_count for r in bucket_records)
                avg_latency = sum(r.latency_ms for r in bucket_records) / len(
                    bucket_records
                )
                avg_throughput = sum(
                    r.throughput_tokens_per_sec for r in bucket_records
                ) / len(bucket_records)

                buckets.append(
                    {
                        "timestamp": current_time.isoformat(),
                        "count": len(bucket_records),
                        "total_tokens": total_tokens,
                        "avg_latency_ms": avg_latency,
                        "avg_throughput": avg_throughput,
                    }
                )

            current_time = bucket_end

        return buckets

    def export_records(
        self,
        modality: Optional[Modality] = None,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """Export filtered records as dictionaries."""
        records = self.get_records(modality, operation, user_id, since)
        return [r.to_dict() for r in records]

    def get_summary(self) -> Dict:
        """
        Get comprehensive usage summary.

        Returns:
            Dictionary with overall statistics
        """
        return {
            "total_records": len(self._records),
            "overall_metrics": self.get_metrics(),
            "by_modality": self.get_metrics_by_modality(),
            "by_operation": self.get_metrics_by_operation(),
            "by_user": self.get_metrics_by_user(),
            "retention_hours": self.retention_hours,
        }

    def reset(self):
        """Clear all usage records."""
        self._records.clear()

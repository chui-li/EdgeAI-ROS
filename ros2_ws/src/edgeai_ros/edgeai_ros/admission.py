from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdmissionDecision:
    action: str  # accept, degrade, drop, defer, reject
    reason: str


class AdmissionController:
    """Deadline-aware admission controller for real-time AI workloads."""

    def __init__(
        self,
        deadline_ms: float,
        max_queue: int = 4,
        reject_threshold: float = 2.5,
        overload_rho: float = 1.0,
        drop_stale_ms: float = 0.0,
    ):
        self.deadline_ms = deadline_ms
        self.max_queue = max(1, max_queue)
        self.reject_threshold = reject_threshold
        self.overload_rho = float(overload_rho)
        self.drop_stale_ms = float(drop_stale_ms)

    def decide(
        self,
        predicted_latency_ms: float,
        queue_size: int,
        priority: str = "normal",
        utilization_rho: float = 0.0,
        stale_frame_age_ms: float = 0.0,
        queue_policy: str = "fifo",
    ) -> AdmissionDecision:
        ratio = predicted_latency_ms / max(1e-6, self.deadline_ms)
        queue_policy = str(queue_policy).lower()

        if self.drop_stale_ms > 0.0 and stale_frame_age_ms > self.drop_stale_ms:
            return AdmissionDecision("drop", f"stale frame age={stale_frame_age_ms:.1f}ms")

        if queue_policy in {"latest_only", "latest-only"} and queue_size > 0:
            return AdmissionDecision("drop", f"latest-only queue_size={queue_size}")

        if queue_policy in {"deadline_drop", "deadline-aware-drop"} and ratio > 1.0:
            return AdmissionDecision("drop", f"deadline-aware drop ratio={ratio:.2f}")

        if utilization_rho >= self.overload_rho and ratio > 1.0:
            return AdmissionDecision("degrade", f"rho={utilization_rho:.2f} ratio={ratio:.2f}")

        if priority == "high":
            if ratio <= self.reject_threshold:
                return AdmissionDecision("accept" if ratio <= 1.0 else "degrade", f"high priority ratio={ratio:.2f}")
            return AdmissionDecision("degrade", f"high priority forced degrade ratio={ratio:.2f}")

        if queue_size >= self.max_queue:
            return AdmissionDecision("drop", f"queue full {queue_size}/{self.max_queue}")
        if ratio <= 1.0:
            return AdmissionDecision("accept", f"predicted within deadline ratio={ratio:.2f}")
        if ratio <= self.reject_threshold:
            return AdmissionDecision("degrade", f"predicted exceeds deadline ratio={ratio:.2f}")
        return AdmissionDecision("defer", f"predicted too slow ratio={ratio:.2f}")

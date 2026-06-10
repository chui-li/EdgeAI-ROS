from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .monitor import OSState
from .predictor import EWMALatencyPredictor, RuntimeConfig


@dataclass
class SchedulerDecision:
    config: RuntimeConfig
    pressure_score: float
    predicted_latency_ms: float
    policy: str
    action: str
    reason: str


class PredictiveScheduler:
    """Predictive OS-aware scheduler with hard-deadline-aware utility.

    Policies:
    - static_large: always choose the highest quality config
    - static_small: always choose the lightest config
    - rule_adaptive: reactive feedback policy
    - predictive_adaptive: EWMA prediction + feasibility filter + utility

    Major fix in this version:
    The previous predictive policy over-weighted quality and under-penalized
    deadline violations. It often selected the largest config even when it
    missed the deadline. This version uses a two-stage rule:

    1. Feasibility filter:
       Prefer configs whose predicted latency <= deadline * safety_margin.

    2. Utility selection:
       Among feasible configs, choose the one with the best quality-aware utility.
       If no config is feasible, choose the config with the lowest predicted
       latency instead of blindly choosing the highest quality config.
    """

    def __init__(
        self,
        configs: List[RuntimeConfig],
        predictor: Optional[EWMALatencyPredictor] = None,
        policy: str = "predictive_adaptive",
        deadline_ms: float = 33.0,
        latency_weight: float = 1.25,
        memory_weight: float = 0.20,
        pressure_weight: float = 0.20,
        quality_weight: float = 0.85,
        safety_margin: float = 0.95,
    ):
        if not configs:
            raise ValueError("configs cannot be empty")

        self.configs = sorted(configs, key=lambda c: c.level)
        self.predictor = predictor or EWMALatencyPredictor()
        self.policy = policy
        self.deadline_ms = float(deadline_ms)

        self.latency_weight = float(latency_weight)
        self.memory_weight = float(memory_weight)
        self.pressure_weight = float(pressure_weight)
        self.quality_weight = float(quality_weight)
        self.safety_margin = float(safety_margin)

        self.current_index = len(self.configs) - 1
        self.last_latency_ms: Optional[float] = None
        self.good_count = 0
        self.bad_count = 0

    def compute_pressure(self, state: OSState, queue_size: int = 0, max_queue: int = 4) -> float:
        queue_pressure = min(100.0, 100.0 * queue_size / max(1, max_queue))
        pf_pressure = min(100.0, state.page_faults_delta / 2000.0 * 100.0)
        ctx_pressure = min(100.0, state.ctx_switches_delta / 200.0 * 100.0)
        io_pressure = min(
            100.0,
            (state.io_read_delta + state.io_write_delta) / (32 * 1024 * 1024) * 100.0,
        )

        latency_ratio = 0.0
        if self.last_latency_ms is not None:
            latency_ratio = min(100.0, 100.0 * self.last_latency_ms / max(1e-6, self.deadline_ms))

        return float(
            0.24 * state.cpu_percent
            + 0.20 * state.memory_percent
            + 0.25 * latency_ratio
            + 0.14 * queue_pressure
            + 0.10 * pf_pressure
            + 0.07 * ctx_pressure
            + 0.05 * io_pressure
        )

    def update_feedback(self, config: RuntimeConfig, observed_latency_ms: float):
        self.last_latency_ms = float(observed_latency_ms)
        self.predictor.update(config, observed_latency_ms)

        if observed_latency_ms > self.deadline_ms:
            self.bad_count += 1
            self.good_count = 0
        elif observed_latency_ms < self.deadline_ms * 0.65:
            self.good_count += 1
            self.bad_count = 0
        else:
            self.good_count = 0
            self.bad_count = 0

    def _deadline_penalty(self, pred: float) -> float:
        ratio = pred / max(1e-6, self.deadline_ms)
        if ratio <= 1.0:
            return 0.0
        # Quadratic penalty makes hard misses very expensive.
        return (ratio - 1.0) ** 2 + 1.50 * (ratio - 1.0)

    def _predictive_score(self, cfg: RuntimeConfig, pred: float, state: OSState, pressure: float) -> float:
        deadline_penalty = self._deadline_penalty(pred)
        memory_penalty = max(0.0, state.memory_percent / 100.0 - 0.70)
        pressure_penalty = pressure / 100.0

        utility = (
            self.quality_weight * cfg.quality_score
            - self.latency_weight * deadline_penalty
            - self.memory_weight * memory_penalty
            - self.pressure_weight * pressure_penalty
        )

        # Bonus for safely satisfying the deadline. The lower the prediction,
        # the larger the slack bonus, but it is bounded to avoid always picking
        # the smallest config.
        if pred <= self.deadline_ms:
            slack = (self.deadline_ms - pred) / max(1e-6, self.deadline_ms)
            utility += 0.20 * min(1.0, max(0.0, slack))

        return float(utility)

    def select(self, state: OSState, queue_size: int = 0, max_queue: int = 4, action: str = "accept") -> SchedulerDecision:
        pressure = self.compute_pressure(state, queue_size, max_queue)

        if self.policy == "static_small":
            cfg = self.configs[0]
            pred = self.predictor.predict(cfg, pressure)
            return SchedulerDecision(cfg, pressure, pred, self.policy, action, "static_small")

        if self.policy == "static_large":
            cfg = self.configs[-1]
            pred = self.predictor.predict(cfg, pressure)
            return SchedulerDecision(cfg, pressure, pred, self.policy, action, "static_large")

        force_degrade = action in {"degrade", "defer"}

        if self.policy == "rule_adaptive":
            if force_degrade or pressure > 75 or self.bad_count >= 1:
                self.current_index = max(0, self.current_index - 1)
            elif pressure < 45 and self.good_count >= 6 and queue_size == 0:
                self.current_index = min(len(self.configs) - 1, self.current_index + 1)
                self.good_count = 0

            cfg = self.configs[self.current_index]
            pred = self.predictor.predict(cfg, pressure)
            return SchedulerDecision(
                cfg,
                pressure,
                pred,
                self.policy,
                action,
                f"rule pressure={pressure:.1f}, bad={self.bad_count}, good={self.good_count}",
            )

        # predictive_adaptive
        evaluated = []
        for cfg in self.configs:
            pred = self.predictor.predict(cfg, pressure)
            utility = self._predictive_score(cfg, pred, state, pressure)
            feasible = pred <= self.deadline_ms * self.safety_margin
            evaluated.append((cfg, pred, utility, feasible))

        feasible_items = [x for x in evaluated if x[3]]

        if feasible_items:
            # Choose highest utility among feasible configs.
            best_cfg, best_pred, best_utility, _ = max(feasible_items, key=lambda x: x[2])
            selected_reason = "feasible-best-utility"
        else:
            # If everything is predicted to miss, pick fastest config.
            # This protects real-time deadline better than quality-dominated utility.
            best_cfg, best_pred, best_utility, _ = min(evaluated, key=lambda x: x[1])
            selected_reason = "no-feasible-pick-fastest"

        if force_degrade:
            current_best_index = self.configs.index(best_cfg)
            cap_index = max(0, current_best_index - 1)
            capped_cfg = self.configs[cap_index]
            capped_pred = self.predictor.predict(capped_cfg, pressure)
            if capped_cfg.level < best_cfg.level:
                best_cfg = capped_cfg
                best_pred = capped_pred
                best_utility = self._predictive_score(capped_cfg, capped_pred, state, pressure)
                selected_reason += "+admission-degrade"

        # Reactive safety: if the latest observed latency missed the deadline,
        # prevent immediate jump back to larger configs.
        if self.bad_count >= 1:
            current_best_index = self.configs.index(best_cfg)
            cap_index = max(0, min(current_best_index, self.current_index))
            # If current selected still too high, degrade by one level.
            cap_index = max(0, cap_index - 1)
            capped_cfg = self.configs[cap_index]
            capped_pred = self.predictor.predict(capped_cfg, pressure)
            if capped_cfg.level < best_cfg.level:
                best_cfg = capped_cfg
                best_pred = capped_pred
                best_utility = self._predictive_score(capped_cfg, capped_pred, state, pressure)
                selected_reason += "+reactive-cap"

        self.current_index = self.configs.index(best_cfg)

        reason_parts = []
        for cfg, pred, utility, feasible in evaluated:
            tag = "ok" if feasible else "miss"
            reason_parts.append(f"L{cfg.level}:{tag},pred={pred:.1f},u={utility:.3f}")

        return SchedulerDecision(
            best_cfg,
            pressure,
            float(best_pred),
            self.policy,
            action,
            f"predictive_v2 {selected_reason}; pressure={pressure:.1f}; "
            + " | ".join(reason_parts),
        )

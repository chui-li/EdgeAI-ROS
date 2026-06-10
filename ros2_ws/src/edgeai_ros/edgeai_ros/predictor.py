from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for both ViT and LLM workloads.

    level:
        0 = lightest / fastest config
        larger level = higher quality but usually slower

    quality_score:
        A proxy score used by the scheduler. It does not represent real
        ImageNet accuracy or LLM quality. It is only a runtime utility score.
    """
    level: int
    model_name: str
    image_size: int = 224
    context_length: int = 1024
    max_new_tokens: int = 64
    quality_score: float = 1.0
    device: str = "auto"

    def key(self) -> Tuple:
        return (
            self.level,
            self.model_name,
            self.image_size,
            self.device,
            self.context_length,
            self.max_new_tokens,
        )


class EWMALatencyPredictor:
    """Lightweight online latency predictor.

    This predictor intentionally remains simple and explainable for an OS
    course project. It maintains:
    - config-specific EWMA latency
    - global EWMA latency
    - fallback heuristic for unseen configs

    Important fix in this version:
    When only a large config has been observed, the old predictor used the
    same global latency for all unseen configs. That made the scheduler think
    small configs were almost as slow as large configs, so predictive scheduling
    kept choosing the largest model. We now scale unseen configs using their
    quality_score so smaller configs are estimated to be cheaper before they
    have measurements.
    """

    def __init__(self, alpha: float = 0.35, pressure_weight: float = 0.20):
        self.alpha = alpha
        self.pressure_weight = pressure_weight
        self.latency_by_config: Dict[Tuple, float] = {}
        self.global_ewma = None
        self.samples = defaultdict(int)

    def update(self, config: RuntimeConfig, observed_latency_ms: float):
        observed_latency_ms = float(max(0.0, observed_latency_ms))
        k = config.key()

        if k not in self.latency_by_config:
            self.latency_by_config[k] = observed_latency_ms
        else:
            prev = self.latency_by_config[k]
            self.latency_by_config[k] = (
                self.alpha * observed_latency_ms + (1.0 - self.alpha) * prev
            )

        self.samples[k] += 1

        if self.global_ewma is None:
            self.global_ewma = observed_latency_ms
        else:
            self.global_ewma = (
                self.alpha * observed_latency_ms + (1.0 - self.alpha) * self.global_ewma
            )

    def _unseen_scale(self, config: RuntimeConfig) -> float:
        """Estimate relative runtime cost for an unseen config.

        quality_score is normally:
        - ViT: 0.72 / 0.86 / 1.00
        - LLM: 0.68 / 0.84 / 1.00

        The quadratic mapping makes small configs clearly cheaper. This is
        important for predictive adaptation before every config has been tried.
        """
        q = max(0.0, min(1.0, float(config.quality_score)))
        return 0.10 + 0.90 * (q ** 2)

    def predict(self, config: RuntimeConfig, pressure_score: float, fallback_latency_ms: float = 50.0) -> float:
        k = config.key()

        if k in self.latency_by_config:
            base = self.latency_by_config[k]
        elif self.global_ewma is not None:
            # Heuristic for unseen configs based on relative quality/cost.
            base = self.global_ewma * self._unseen_scale(config)
        else:
            # No observation yet. Use fallback scaled by cost.
            base = fallback_latency_ms * self._unseen_scale(config)

        pressure_multiplier = 1.0 + self.pressure_weight * max(0.0, pressure_score) / 100.0
        return float(base * pressure_multiplier)

    def has_sample(self, config: RuntimeConfig) -> bool:
        return self.samples[config.key()] > 0

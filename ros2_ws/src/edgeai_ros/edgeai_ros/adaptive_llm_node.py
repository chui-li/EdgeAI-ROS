from __future__ import annotations

import time

import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import String

from .admission import AdmissionController
from .llm_worker import generate_once, load_llm
from .monitor import OSMonitor
from .predictor import EWMALatencyPredictor, RuntimeConfig
from .scheduler import PredictiveScheduler


def kv_line(**items):
    return ", ".join(f"{key}={value}" for key, value in items.items())


def build_llm_configs(model_name: str):
    return [
        RuntimeConfig(level=0, model_name=model_name, context_length=512, max_new_tokens=32, quality_score=0.68),
        RuntimeConfig(level=1, model_name=model_name, context_length=1024, max_new_tokens=64, quality_score=0.84),
        RuntimeConfig(level=2, model_name=model_name, context_length=2048, max_new_tokens=128, quality_score=1.00),
    ]


class AdaptiveLLMNode(Node):
    """ROS 2 LLM inference node with OS-aware context/token adaptation."""

    def __init__(self):
        super().__init__("adaptive_llm_node")
        self.declare_parameter("prompt_topic", "/llm_prompt")
        self.declare_parameter("metrics_topic", "/llm_metrics")
        self.declare_parameter("decision_topic", "/llm_scheduler_decision")
        self.declare_parameter("policy", "predictive_adaptive")
        self.declare_parameter("model", "sshleifer/tiny-gpt2")
        self.declare_parameter("deadline_ms", 80.0)
        self.declare_parameter("priority", "normal")
        self.declare_parameter("device", "auto")
        self.declare_parameter("local_files_only", False)
        self.declare_parameter("max_queue", 1)

        self.prompt_topic = self.get_parameter("prompt_topic").value
        self.metrics_topic = self.get_parameter("metrics_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value
        self.policy = self.get_parameter("policy").value
        self.model_name = self.get_parameter("model").value
        self.deadline_ms = float(self.get_parameter("deadline_ms").value)
        self.priority = self.get_parameter("priority").value
        self.max_queue = int(self.get_parameter("max_queue").value)

        device_name = self.get_parameter("device").value
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            self.get_logger().warning("CUDA requested but unavailable; using CPU")
            device_name = "cpu"
        self.device = torch.device(device_name)

        self.monitor = OSMonitor()
        self.predictor = EWMALatencyPredictor()
        self.scheduler = PredictiveScheduler(
            configs=build_llm_configs(self.model_name),
            predictor=self.predictor,
            policy=self.policy,
            deadline_ms=self.deadline_ms,
            latency_weight=1.35,
            quality_weight=0.80,
        )
        self.admission = AdmissionController(self.deadline_ms, max_queue=self.max_queue)
        self.model, self.tokenizer, self.fallback = load_llm(
            self.model_name,
            self.device,
            local_files_only=bool(self.get_parameter("local_files_only").value),
        )

        self.sub = self.create_subscription(String, self.prompt_topic, self.prompt_callback, 10)
        self.metrics_pub = self.create_publisher(String, self.metrics_topic, 10)
        self.decision_pub = self.create_publisher(String, self.decision_topic, 10)

        self.request_count = 0
        self.deadline_misses = 0
        self.deferred_requests = 0
        self.in_callback = False

        self.get_logger().info(
            f"Adaptive LLM ready: model={self.model_name}, fallback={self.fallback}, "
            f"policy={self.policy}, device={self.device}, deadline_ms={self.deadline_ms}"
        )

    def _publish(self, pub, data: str):
        msg = String()
        msg.data = data
        pub.publish(msg)

    def prompt_callback(self, msg: String):
        request_start = time.perf_counter()
        self.request_count += 1
        request_id = self.request_count
        queue_size = 1 if self.in_callback else 0
        self.in_callback = True

        try:
            state = self.monitor.sample()
            pressure_preview = self.scheduler.compute_pressure(state, queue_size, self.max_queue)
            pred_for_admission = self.predictor.global_ewma or self.deadline_ms
            pred_for_admission *= 1.0 + 0.2 * pressure_preview / 100.0
            admission_decision = self.admission.decide(pred_for_admission, queue_size, self.priority)
            decision = self.scheduler.select(
                state,
                queue_size=queue_size,
                max_queue=self.max_queue,
                action=admission_decision.action,
            )
            cfg = decision.config
            action = admission_decision.action

            if action in ["reject", "defer", "drop"]:
                self.deferred_requests += 1
                metrics = {
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "tokens_per_sec": 0.0,
                    "output_tokens": 0,
                    "prompt_words": len(msg.data.split()),
                }
                deadline_miss = 0
            else:
                metrics = generate_once(
                    self.model,
                    self.tokenizer,
                    self.fallback,
                    msg.data,
                    cfg.context_length,
                    cfg.max_new_tokens,
                    self.device,
                )
                self.scheduler.update_feedback(cfg, metrics["tpot_ms"])
                deadline_miss = int(metrics["tpot_ms"] > self.deadline_ms)
                self.deadline_misses += deadline_miss

            e2e_ms = (time.perf_counter() - request_start) * 1000.0
            miss_rate = self.deadline_misses / max(1, request_id)

            result = kv_line(
                request=request_id,
                action=action,
                policy=self.policy,
                level=cfg.level,
                model=cfg.model_name,
                context_length=cfg.context_length,
                max_new_tokens=cfg.max_new_tokens,
                quality_score=f"{cfg.quality_score:.3f}",
                prompt_words=metrics["prompt_words"],
                output_tokens=metrics["output_tokens"],
                ttft_ms=f"{metrics['ttft_ms']:.3f}",
                tpot_ms=f"{metrics['tpot_ms']:.3f}",
                tokens_per_sec=f"{metrics['tokens_per_sec']:.3f}",
                e2e_ms=f"{e2e_ms:.3f}",
                predicted_latency_ms=f"{decision.predicted_latency_ms:.3f}",
                deadline_ms=f"{self.deadline_ms:.3f}",
                deadline_miss=deadline_miss,
                deadline_miss_rate=f"{miss_rate:.4f}",
                pressure_score=f"{decision.pressure_score:.3f}",
                queue_size=queue_size,
                deferred_requests=self.deferred_requests,
                cpu_percent=f"{state.cpu_percent:.3f}",
                memory_percent=f"{state.memory_percent:.3f}",
                process_rss_mb=f"{state.process_rss_mb:.3f}",
                page_faults_delta=state.page_faults_delta,
                ctx_switches_delta=state.ctx_switches_delta,
                io_read_delta=state.io_read_delta,
                io_write_delta=state.io_write_delta,
                fallback_model=int(self.fallback),
                device=str(self.device),
            )
            sched = kv_line(
                request=request_id,
                action=action,
                policy=self.policy,
                level=cfg.level,
                model=cfg.model_name,
                context_length=cfg.context_length,
                max_new_tokens=cfg.max_new_tokens,
                pressure_score=f"{decision.pressure_score:.3f}",
                predicted_latency_ms=f"{decision.predicted_latency_ms:.3f}",
                reason=decision.reason.replace(",", ";"),
                admission_reason=admission_decision.reason.replace(",", ";"),
            )

            self._publish(self.metrics_pub, result)
            self._publish(self.decision_pub, sched)
            self.get_logger().info(result)
        finally:
            self.in_callback = False


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveLLMNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


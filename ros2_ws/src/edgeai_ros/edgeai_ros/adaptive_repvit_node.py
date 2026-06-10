from __future__ import annotations

import time
import os

import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .admission import AdmissionController
from .monitor import OSMonitor
from .predictor import EWMALatencyPredictor, RuntimeConfig
from .repvit_backend import AdaptiveModelCache, image_msg_to_bgr, preprocess_bgr
from .scheduler import PredictiveScheduler


def _split_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _qos_profile(depth: int, reliability: str):
    rel = ReliabilityPolicy.RELIABLE
    if str(reliability).lower() in {"best_effort", "besteffort", "best-effort"}:
        rel = ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=max(1, int(depth)), reliability=rel)


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def _set_os_scheduling(cpu_affinity: str, nice_delta: int, logger):
    if str(cpu_affinity).lower() in {"none", "null", "__none__"}:
        cpu_affinity = ""
    if cpu_affinity:
        try:
            cores = {int(x.strip()) for x in cpu_affinity.split(",") if x.strip()}
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, cores)
                logger.info(f"Set CPU affinity to {sorted(cores)}")
        except Exception as exc:
            logger.warning(f"Failed to set CPU affinity '{cpu_affinity}': {exc}")
    if nice_delta:
        try:
            os.nice(int(nice_delta))
            logger.info(f"Adjusted process nice by {nice_delta}")
        except Exception as exc:
            logger.warning(f"Failed to adjust process nice by {nice_delta}: {exc}")


def _msg_stamp_to_perf_age_ms(msg: Image):
    stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
    if stamp_sec <= 0.0:
        return 0.0
    return max(0.0, time.time() * 1000.0 - stamp_sec * 1000.0)


def _parse_frame_id(frame_id: str):
    parts = str(frame_id).split("|")
    meta = {"image_name": parts[0] if parts else str(frame_id)}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _float_meta(meta, key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except Exception:
        return default


def _gpu_memory_mb(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return (
            float(torch.cuda.memory_allocated(idx) / 1024 / 1024),
            float(torch.cuda.memory_reserved(idx) / 1024 / 1024),
        )
    except Exception:
        return 0.0, 0.0


def build_configs(models: str, image_sizes: str, quality_scores: str, devices: str = ""):
    model_names = _split_csv(models)
    sizes = [int(x) for x in _split_csv(image_sizes)]
    qualities = [float(x) for x in _split_csv(quality_scores)]
    device_names = [] if str(devices).lower() in {"none", "null", "__none__"} else _split_csv(devices)
    if not model_names:
        model_names = ["repvit_m0_9"]
    configs = []
    n = max(len(model_names), len(sizes), len(qualities))
    for i in range(n):
        name = model_names[min(i, len(model_names) - 1)]
        size = sizes[min(i, len(sizes) - 1)] if sizes else 224
        quality = qualities[min(i, len(qualities) - 1)] if qualities else 1.0
        device = device_names[min(i, len(device_names) - 1)] if device_names else "auto"
        configs.append(
            RuntimeConfig(
                level=i,
                model_name=name,
                image_size=size,
                quality_score=quality,
                device=_resolve_device(device),
            )
        )
    return configs


def kv_line(**items):
    return ", ".join(f"{key}={value}" for key, value in items.items())


class AdaptiveRepViTNode(Node):
    """ROS 2 image inference node with OS-aware adaptive scheduling."""

    def __init__(self):
        super().__init__("adaptive_repvit_node")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("latency_topic", "/repvit_latency")
        self.declare_parameter("os_state_topic", "/edgeai_os_state")
        self.declare_parameter("decision_topic", "/edgeai_scheduler_decision")
        self.declare_parameter("policy", "predictive_adaptive")
        self.declare_parameter("models", "repvit_m0_9,repvit_m0_9,repvit_m0_9")
        self.declare_parameter("image_sizes", "160,192,224")
        self.declare_parameter("quality_scores", "0.72,0.86,1.0")
        self.declare_parameter("devices", "")
        self.declare_parameter("deadline_ms", 33.0)
        self.declare_parameter("max_queue", 4)
        self.declare_parameter("priority", "normal")
        self.declare_parameter("device", "auto")
        self.declare_parameter("cache_size", 2)
        self.declare_parameter("repvit_path", "")
        self.declare_parameter("checkpoint_dir", "/workspace/EdgeAI-ROS/checkpoints")
        self.declare_parameter("use_timm_fallback", True)
        self.declare_parameter("preload_models", False)
        self.declare_parameter("publish_every_frame", True)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("qos_reliability", "reliable")
        self.declare_parameter("queue_policy", "fifo")
        self.declare_parameter("cpu_affinity", "")
        self.declare_parameter("nice_delta", 0)
        self.declare_parameter("drop_stale_ms", 0.0)
        self.declare_parameter("overload_rho", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.latency_topic = self.get_parameter("latency_topic").value
        self.os_state_topic = self.get_parameter("os_state_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value
        self.policy = self.get_parameter("policy").value
        self.deadline_ms = float(self.get_parameter("deadline_ms").value)
        self.max_queue = int(self.get_parameter("max_queue").value)
        self.priority = self.get_parameter("priority").value
        self.publish_every_frame = bool(self.get_parameter("publish_every_frame").value)
        self.queue_policy = self.get_parameter("queue_policy").value
        qos = _qos_profile(
            int(self.get_parameter("qos_depth").value),
            self.get_parameter("qos_reliability").value,
        )
        _set_os_scheduling(
            self.get_parameter("cpu_affinity").value,
            int(self.get_parameter("nice_delta").value),
            self.get_logger(),
        )

        device_name = self.get_parameter("device").value
        resolved_default_device = _resolve_device(device_name)
        if device_name == "cuda" and resolved_default_device == "cpu":
            self.get_logger().warning("CUDA requested but unavailable; using CPU")
        self.device = torch.device(resolved_default_device)

        configs = build_configs(
            self.get_parameter("models").value,
            self.get_parameter("image_sizes").value,
            self.get_parameter("quality_scores").value,
            self.get_parameter("devices").value,
        )
        self.monitor = OSMonitor()
        self.predictor = EWMALatencyPredictor()
        self.scheduler = PredictiveScheduler(
            configs=configs,
            predictor=self.predictor,
            policy=self.policy,
            deadline_ms=self.deadline_ms,
        )
        self.admission = AdmissionController(
            self.deadline_ms,
            max_queue=self.max_queue,
            overload_rho=float(self.get_parameter("overload_rho").value),
            drop_stale_ms=float(self.get_parameter("drop_stale_ms").value),
        )
        self.model_cache = AdaptiveModelCache(
            device=self.device,
            cache_size=int(self.get_parameter("cache_size").value),
            repvit_path=self.get_parameter("repvit_path").value,
            checkpoint_dir=self.get_parameter("checkpoint_dir").value,
            use_timm_fallback=bool(self.get_parameter("use_timm_fallback").value),
            logger=self.get_logger(),
        )
        if bool(self.get_parameter("preload_models").value):
            self.get_logger().info("Preloading configured RepViT models before subscribing")
            for cfg in configs:
                cfg_device = torch.device(_resolve_device(cfg.device if cfg.device != "auto" else str(self.device)))
                self.model_cache.get(cfg.model_name, cfg_device)

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.latency_pub = self.create_publisher(String, self.latency_topic, qos)
        self.os_pub = self.create_publisher(String, self.os_state_topic, qos)
        self.decision_pub = self.create_publisher(String, self.decision_topic, qos)

        self.frame_count = 0
        self.deadline_misses = 0
        self.dropped_frames = 0
        self.deferred_frames = 0
        self.accepted_frames = 0
        self.model_switch_count = 0
        self.deadline_miss_after_switch = 0
        self.last_config_key = None
        self.last_receive_epoch_ms = None
        self.last_service_ms = None
        self.in_callback = False

        self.get_logger().info(
            f"Adaptive RepViT ready: policy={self.policy}, device={self.device}, "
            f"deadline_ms={self.deadline_ms}, configs={len(configs)}, queue_policy={self.queue_policy}"
        )

    def _publish_string(self, pub, data: str):
        msg = String()
        msg.data = data
        pub.publish(msg)

    def image_callback(self, msg: Image):
        callback_start = time.perf_counter()
        callback_receive_epoch_ms = time.time() * 1000.0
        frame_meta = _parse_frame_id(msg.header.frame_id)
        image_name = frame_meta.get("image_name", msg.header.frame_id)
        image_read_start_epoch_ms = _float_meta(frame_meta, "image_read_start_epoch_ms")
        image_read_end_epoch_ms = _float_meta(frame_meta, "image_read_end_epoch_ms")
        image_publish_epoch_ms = _float_meta(frame_meta, "image_publish_epoch_ms")
        publisher_seq = frame_meta.get("pub_seq", "")
        image_load_ms = max(0.0, image_read_end_epoch_ms - image_read_start_epoch_ms) if image_read_end_epoch_ms else 0.0
        publish_overhead_ms = (
            max(0.0, callback_receive_epoch_ms - image_publish_epoch_ms) if image_publish_epoch_ms else 0.0
        )
        capture_age_at_receive_ms = _msg_stamp_to_perf_age_ms(msg)
        self.frame_count += 1
        frame_id = self.frame_count
        queue_size = 1 if self.in_callback else 0
        self.in_callback = True

        try:
            state = self.monitor.sample()
            arrival_interval_ms = 0.0
            arrival_rate_hz = 0.0
            if self.last_receive_epoch_ms is not None:
                arrival_interval_ms = max(0.0, callback_receive_epoch_ms - self.last_receive_epoch_ms)
                if arrival_interval_ms > 0:
                    arrival_rate_hz = 1000.0 / arrival_interval_ms
            service_rate_hz = 1000.0 / self.last_service_ms if self.last_service_ms and self.last_service_ms > 0 else 0.0
            utilization_rho = arrival_rate_hz / service_rate_hz if service_rate_hz > 0 else 0.0
            pre_decision = self.scheduler.select(
                state,
                queue_size=queue_size,
                max_queue=self.max_queue,
                action="accept",
            )
            admission_decision = self.admission.decide(
                pre_decision.predicted_latency_ms,
                queue_size,
                self.priority,
                utilization_rho=utilization_rho,
                stale_frame_age_ms=capture_age_at_receive_ms,
                queue_policy=self.queue_policy,
            )
            if admission_decision.action == "degrade":
                decision = self.scheduler.select(
                    state,
                    queue_size=queue_size,
                    max_queue=self.max_queue,
                    action=admission_decision.action,
                )
            else:
                pre_decision.action = admission_decision.action
                decision = pre_decision
            cfg = decision.config
            cfg_key = cfg.key()
            model_switched = int(self.last_config_key is not None and cfg_key != self.last_config_key)
            if model_switched:
                self.model_switch_count += 1

            action = admission_decision.action
            pred_class = -1
            infer_ms = 0.0
            preprocess_ms = 0.0
            postprocess_ms = 0.0
            model_get_ms = 0.0
            model_load_latency_ms = 0.0
            model_cache_hit = 0
            model_cache_miss = 0
            fallback_model = 0
            model_source = "none"
            runtime_device = str(self.device)
            deadline_miss = 0
            gpu_memory_allocated_mb = 0.0
            gpu_memory_reserved_mb = 0.0
            memory_rss_after_switch_mb = 0.0

            if action in ["drop", "defer", "reject"] or (
                self.queue_policy == "deadline_drop" and decision.predicted_latency_ms > self.deadline_ms
            ):
                if action == "accept":
                    action = "drop"
                if action == "defer":
                    self.deferred_frames += 1
                else:
                    self.dropped_frames += 1
            else:
                cfg_device = torch.device(_resolve_device(cfg.device if cfg.device != "auto" else str(self.device)))
                model_get_start = time.perf_counter()
                model_result = self.model_cache.get(cfg.model_name, cfg_device)
                model_get_ms = (time.perf_counter() - model_get_start) * 1000.0
                model_load_latency_ms = model_result.load_latency_ms
                model_cache_hit = model_result.cache_hit
                model_cache_miss = int(not bool(model_cache_hit))
                fallback_model = model_result.fallback_model
                model_source = model_result.source
                runtime_device = str(model_result.device)
                preprocess_start = time.perf_counter()
                frame = image_msg_to_bgr(msg)
                x = preprocess_bgr(frame, cfg.image_size, model_result.device)
                preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
                if model_result.device.type == "cuda":
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()
                y = model_result.model(x)
                if model_result.device.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                post_start = time.perf_counter()
                try:
                    pred_class = int(torch.argmax(y, dim=1).item())
                except Exception:
                    pred_class = -1
                postprocess_ms = (time.perf_counter() - post_start) * 1000.0
                self.scheduler.update_feedback(cfg, infer_ms)
                deadline_miss = int(infer_ms > self.deadline_ms)
                self.deadline_misses += deadline_miss
                if model_switched and deadline_miss:
                    self.deadline_miss_after_switch += 1
                self.accepted_frames += 1
                gpu_memory_allocated_mb, gpu_memory_reserved_mb = _gpu_memory_mb(model_result.device)
                memory_rss_after_switch_mb = state.process_rss_mb

            e2e_ms = (time.perf_counter() - callback_start) * 1000.0
            self.last_service_ms = e2e_ms
            self.last_receive_epoch_ms = callback_receive_epoch_ms
            self.last_config_key = cfg_key
            freshness_ms = capture_age_at_receive_ms + e2e_ms
            non_model_overhead_ms = max(0.0, e2e_ms - infer_ms)
            miss_rate = self.deadline_misses / max(1, frame_id)
            effective_fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0.0

            result = kv_line(
                frame=frame_id,
                publisher_seq=publisher_seq,
                image=image_name,
                action=action,
                policy=self.policy,
                level=cfg.level,
                model=cfg.model_name,
                image_size=cfg.image_size,
                config_device=cfg.device,
                quality_score=f"{cfg.quality_score:.3f}",
                pred_class=pred_class,
                image_read_start_epoch_ms=f"{image_read_start_epoch_ms:.3f}",
                image_read_end_epoch_ms=f"{image_read_end_epoch_ms:.3f}",
                image_publish_epoch_ms=f"{image_publish_epoch_ms:.3f}",
                callback_receive_epoch_ms=f"{callback_receive_epoch_ms:.3f}",
                image_load_ms=f"{image_load_ms:.3f}",
                publish_overhead_ms=f"{publish_overhead_ms:.3f}",
                preprocess_ms=f"{preprocess_ms:.3f}",
                infer_ms=f"{infer_ms:.3f}",
                postprocess_ms=f"{postprocess_ms:.3f}",
                model_get_ms=f"{model_get_ms:.3f}",
                model_load_latency_ms=f"{model_load_latency_ms:.3f}",
                model_cache_hit=model_cache_hit,
                model_cache_miss=model_cache_miss,
                model_switched=model_switched,
                model_switch_count=self.model_switch_count,
                memory_rss_after_switch_mb=f"{memory_rss_after_switch_mb:.3f}",
                gpu_memory_allocated_mb=f"{gpu_memory_allocated_mb:.3f}",
                gpu_memory_reserved_mb=f"{gpu_memory_reserved_mb:.3f}",
                deadline_miss_after_switch=self.deadline_miss_after_switch,
                capture_age_at_receive_ms=f"{capture_age_at_receive_ms:.3f}",
                e2e_ms=f"{e2e_ms:.3f}",
                freshness_ms=f"{freshness_ms:.3f}",
                non_model_overhead_ms=f"{non_model_overhead_ms:.3f}",
                inference_ratio=f"{(infer_ms / e2e_ms if e2e_ms > 0 else 0.0):.4f}",
                preprocess_ratio=f"{(preprocess_ms / e2e_ms if e2e_ms > 0 else 0.0):.4f}",
                communication_ratio=f"{(publish_overhead_ms / e2e_ms if e2e_ms > 0 else 0.0):.4f}",
                predicted_latency_ms=f"{decision.predicted_latency_ms:.3f}",
                deadline_ms=f"{self.deadline_ms:.3f}",
                deadline_miss=deadline_miss,
                deadline_miss_rate=f"{miss_rate:.4f}",
                pressure_score=f"{decision.pressure_score:.3f}",
                queue_length=queue_size,
                queue_size=queue_size,
                dropped_frames=self.dropped_frames,
                deferred_frames=self.deferred_frames,
                accepted_frames=self.accepted_frames,
                stale_frame_age_ms=f"{capture_age_at_receive_ms:.3f}",
                arrival_interval_ms=f"{arrival_interval_ms:.3f}",
                arrival_rate_hz=f"{arrival_rate_hz:.3f}",
                service_rate_hz=f"{service_rate_hz:.3f}",
                utilization_rho=f"{utilization_rho:.3f}",
                effective_fps=f"{effective_fps:.3f}",
                cpu_percent=f"{state.cpu_percent:.3f}",
                memory_percent=f"{state.memory_percent:.3f}",
                process_rss_mb=f"{state.process_rss_mb:.3f}",
                page_faults_delta=state.page_faults_delta,
                ctx_switches_delta=state.ctx_switches_delta,
                io_read_delta=state.io_read_delta,
                io_write_delta=state.io_write_delta,
                cpu_num=state.cpu_num,
                cpu_migration_delta=state.cpu_migration_delta,
                cpu_percent_per_core=state.cpu_percent_per_core,
                fallback_model=fallback_model,
                model_source=model_source,
                device=runtime_device,
                queue_policy=self.queue_policy,
                result_publish_epoch_ms=f"{time.time() * 1000.0:.3f}",
            )

            os_state = kv_line(**state.to_dict())
            sched = kv_line(
                frame=frame_id,
                action=action,
                policy=self.policy,
                level=cfg.level,
                model=cfg.model_name,
                image_size=cfg.image_size,
                config_device=cfg.device,
                pressure_score=f"{decision.pressure_score:.3f}",
                predicted_latency_ms=f"{decision.predicted_latency_ms:.3f}",
                reason=decision.reason.replace(",", ";"),
                admission_reason=admission_decision.reason.replace(",", ";"),
            )

            self._publish_string(self.latency_pub, result)
            if self.publish_every_frame:
                self._publish_string(self.os_pub, os_state)
                self._publish_string(self.decision_pub, sched)
            self.get_logger().info(result)
        finally:
            self.in_callback = False


def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveRepViTNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

from __future__ import annotations

import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


FIELDS = [
    "timestamp",
    "frame",
    "request",
    "image",
    "action",
    "policy",
    "level",
    "model",
    "image_size",
    "config_device",
    "context_length",
    "max_new_tokens",
    "quality_score",
    "pred_class",
    "publisher_seq",
    "image_read_start_epoch_ms",
    "image_read_end_epoch_ms",
    "image_publish_epoch_ms",
    "callback_receive_epoch_ms",
    "result_publish_epoch_ms",
    "logger_receive_epoch_ms",
    "logger_write_start_epoch_ms",
    "logger_write_complete_epoch_ms",
    "image_load_ms",
    "publish_overhead_ms",
    "logger_receive_ms",
    "logging_ms",
    "prompt_words",
    "output_tokens",
    "ttft_ms",
    "tpot_ms",
    "tokens_per_sec",
    "preprocess_ms",
    "infer_ms",
    "postprocess_ms",
    "model_get_ms",
    "model_load_latency_ms",
    "model_cache_hit",
    "model_cache_miss",
    "model_switched",
    "model_switch_count",
    "memory_rss_after_switch_mb",
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "deadline_miss_after_switch",
    "capture_age_at_receive_ms",
    "e2e_ms",
    "freshness_ms",
    "non_model_overhead_ms",
    "inference_ratio",
    "preprocess_ratio",
    "communication_ratio",
    "predicted_latency_ms",
    "deadline_ms",
    "deadline_miss",
    "deadline_miss_rate",
    "pressure_score",
    "queue_length",
    "queue_size",
    "dropped_frames",
    "deferred_requests",
    "deferred_frames",
    "accepted_frames",
    "stale_frame_age_ms",
    "arrival_interval_ms",
    "arrival_rate_hz",
    "service_rate_hz",
    "utilization_rho",
    "effective_fps",
    "cpu_percent",
    "memory_percent",
    "process_rss_mb",
    "page_faults_delta",
    "ctx_switches_delta",
    "cpu_num",
    "cpu_migration_delta",
    "cpu_percent_per_core",
    "io_read_delta",
    "io_write_delta",
    "fallback_model",
    "model_source",
    "device",
    "queue_policy",
    "raw_message",
]


def _qos_profile(depth: int, reliability: str):
    rel = ReliabilityPolicy.RELIABLE
    if str(reliability).lower() in {"best_effort", "besteffort", "best-effort"}:
        rel = ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=max(1, int(depth)), reliability=rel)


def parse_kv_line(data: str):
    out = {}
    for part in data.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


class EdgeAILogger(Node):
    def __init__(self):
        super().__init__("edgeai_logger")
        self.declare_parameter("latency_topic", "/repvit_latency")
        self.declare_parameter("csv_path", "/workspace/EdgeAI-ROS/data/results/edgeai_ros_latency.csv")
        self.declare_parameter("append", False)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("qos_reliability", "reliable")

        topic = self.get_parameter("latency_topic").value
        self.csv_path = self.get_parameter("csv_path").value
        append = bool(self.get_parameter("append").value)
        qos = _qos_profile(
            int(self.get_parameter("qos_depth").value),
            self.get_parameter("qos_reliability").value,
        )
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        mode = "a" if append and os.path.exists(self.csv_path) else "w"
        self.file = open(self.csv_path, mode, newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        if mode == "w":
            self.writer.writeheader()

        self.sub = self.create_subscription(String, topic, self.callback, qos)
        self.get_logger().info(f"Logging {topic} to {self.csv_path}")

    def callback(self, msg: String):
        receive_epoch_ms = time.time() * 1000.0
        row = {field: "" for field in FIELDS}
        row.update(parse_kv_line(msg.data))
        row["timestamp"] = receive_epoch_ms / 1000.0
        row["logger_receive_epoch_ms"] = f"{receive_epoch_ms:.3f}"
        try:
            result_publish_epoch_ms = float(row.get("result_publish_epoch_ms") or 0.0)
            if result_publish_epoch_ms > 0.0:
                row["logger_receive_ms"] = f"{max(0.0, receive_epoch_ms - result_publish_epoch_ms):.3f}"
        except Exception:
            pass
        row["raw_message"] = msg.data
        write_start_epoch_ms = time.time() * 1000.0
        row["logger_write_start_epoch_ms"] = f"{write_start_epoch_ms:.3f}"
        row["logger_write_complete_epoch_ms"] = f"{write_start_epoch_ms:.3f}"
        row["logging_ms"] = f"{max(0.0, write_start_epoch_ms - receive_epoch_ms):.3f}"
        self.writer.writerow(row)
        self.file.flush()

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EdgeAILogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

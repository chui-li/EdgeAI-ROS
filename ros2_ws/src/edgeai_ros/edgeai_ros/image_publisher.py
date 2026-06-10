from __future__ import annotations

import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image


def _qos_profile(depth: int, reliability: str):
    rel = ReliabilityPolicy.RELIABLE
    if str(reliability).lower() in {"best_effort", "besteffort", "best-effort"}:
        rel = ReliabilityPolicy.BEST_EFFORT
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=max(1, int(depth)), reliability=rel)


class ImagePublisher(Node):
    def __init__(self):
        super().__init__("edgeai_image_publisher")
        self.declare_parameter("image_dir", "/workspace/EdgeAI-ROS/data/test_images")
        self.declare_parameter("period_sec", 0.5)
        self.declare_parameter("topic", "/image_raw")
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter("qos_reliability", "reliable")

        self.image_dir = self.get_parameter("image_dir").value
        self.topic = self.get_parameter("topic").value
        period_sec = float(self.get_parameter("period_sec").value)
        qos = _qos_profile(
            int(self.get_parameter("qos_depth").value),
            self.get_parameter("qos_reliability").value,
        )

        self.publisher_ = self.create_publisher(Image, self.topic, qos)
        self.image_paths = self._load_paths()
        self.idx = 0
        self.frame_seq = 0

        if not self.image_paths:
            self.get_logger().error(f"No images found in {self.image_dir}")
        else:
            self.get_logger().info(f"Found {len(self.image_paths)} images in {self.image_dir}")

        self.timer = self.create_timer(period_sec, self.publish_image)

    def _load_paths(self):
        if not os.path.isdir(self.image_dir):
            return []
        paths = [
            os.path.join(self.image_dir, name)
            for name in os.listdir(self.image_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return sorted(paths)

    def publish_image(self):
        if not self.image_paths:
            return

        path = self.image_paths[self.idx]
        read_start_epoch_ms = time.time() * 1000.0
        img = cv2.imread(path)
        read_end_epoch_ms = time.time() * 1000.0
        if img is None:
            self.get_logger().warning(f"Failed to read image: {path}")
            self.idx = (self.idx + 1) % len(self.image_paths)
            return

        self.frame_seq += 1
        publish_epoch_ms = time.time() * 1000.0
        msg = Image()
        stamp = Time(seconds=read_start_epoch_ms / 1000.0).to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = (
            f"{os.path.basename(path)}"
            f"|pub_seq={self.frame_seq}"
            f"|image_read_start_epoch_ms={read_start_epoch_ms:.3f}"
            f"|image_read_end_epoch_ms={read_end_epoch_ms:.3f}"
            f"|image_publish_epoch_ms={publish_epoch_ms:.3f}"
        )
        msg.height = int(img.shape[0])
        msg.width = int(img.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(img.shape[1] * 3)
        msg.data = img.tobytes()

        self.publisher_.publish(msg)
        self.get_logger().info(f"Published image: {path}")
        self.idx = (self.idx + 1) % len(self.image_paths)


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

from __future__ import annotations

import rclpy
from rclpy.node import Node

from .runtime import SafeBackgroundLoad


class StressNode(Node):
    def __init__(self):
        super().__init__("edgeai_stress_node")
        self.declare_parameter("mode", "cpu")
        self.declare_parameter("cpu_workers", 2)
        self.declare_parameter("memory_mb", 512)

        self.load = SafeBackgroundLoad()
        mode = self.get_parameter("mode").value
        cpu_workers = int(self.get_parameter("cpu_workers").value)
        memory_mb = int(self.get_parameter("memory_mb").value)
        self.load.start(mode, cpu_workers=cpu_workers, memory_mb=memory_mb)
        self.get_logger().info(
            f"Started stress load: mode={mode}, cpu_workers={cpu_workers}, memory_mb={memory_mb}"
        )

    def destroy_node(self):
        self.load.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StressNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


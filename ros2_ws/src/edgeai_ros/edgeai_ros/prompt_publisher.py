from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PromptPublisher(Node):
    def __init__(self):
        super().__init__("edgeai_prompt_publisher")
        self.declare_parameter("prompt_topic", "/llm_prompt")
        self.declare_parameter("period_sec", 5.0)
        self.declare_parameter(
            "prompt",
            (
                "Explain why operating system resource management matters for "
                "edge AI inference with vision models and language models."
            ),
        )
        self.declare_parameter("prompt_repeat", 1)

        self.publisher_ = self.create_publisher(String, self.get_parameter("prompt_topic").value, 10)
        self.prompt = self._build_prompt()
        self.count = 0
        self.timer = self.create_timer(float(self.get_parameter("period_sec").value), self.publish_prompt)

    def _build_prompt(self):
        prompt = self.get_parameter("prompt").value.strip()
        repeat = max(1, int(self.get_parameter("prompt_repeat").value))
        return " ".join([prompt] * repeat)

    def publish_prompt(self):
        self.count += 1
        msg = String()
        msg.data = self.prompt
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published prompt {self.count}: words={len(self.prompt.split())}")


def main(args=None):
    rclpy.init(args=args)
    node = PromptPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


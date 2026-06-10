from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy = LaunchConfiguration("policy")
    deadline_ms = LaunchConfiguration("deadline_ms")
    image_dir = LaunchConfiguration("image_dir")
    csv_path = LaunchConfiguration("csv_path")
    llm_csv_path = LaunchConfiguration("llm_csv_path")
    repvit_path = LaunchConfiguration("repvit_path")
    checkpoint_dir = LaunchConfiguration("checkpoint_dir")
    enable_llm = LaunchConfiguration("enable_llm")
    llm_deadline_ms = LaunchConfiguration("llm_deadline_ms")

    return LaunchDescription(
        [
            DeclareLaunchArgument("policy", default_value="predictive_adaptive"),
            DeclareLaunchArgument("deadline_ms", default_value="33.0"),
            DeclareLaunchArgument("image_dir", default_value="/workspace/EdgeAI-ROS/data/test_images"),
            DeclareLaunchArgument("csv_path", default_value="/workspace/EdgeAI-ROS/data/results/edgeai_ros_latency.csv"),
            DeclareLaunchArgument("llm_csv_path", default_value="/workspace/EdgeAI-ROS/data/results/edgeai_ros_llm.csv"),
            DeclareLaunchArgument("repvit_path", default_value="/workspace/OS2026/RepViT"),
            DeclareLaunchArgument("checkpoint_dir", default_value="/workspace/EdgeAI-ROS/checkpoints"),
            DeclareLaunchArgument("enable_llm", default_value="false"),
            DeclareLaunchArgument("llm_deadline_ms", default_value="80.0"),
            Node(
                package="edgeai_ros",
                executable="edgeai_logger",
                name="edgeai_logger",
                parameters=[{"csv_path": csv_path}],
                output="screen",
            ),
            Node(
                package="edgeai_ros",
                executable="adaptive_repvit_node",
                name="adaptive_repvit_node",
                parameters=[
                    {
                        "policy": policy,
                        "deadline_ms": deadline_ms,
                        "repvit_path": repvit_path,
                        "checkpoint_dir": checkpoint_dir,
                    }
                ],
                output="screen",
            ),
            Node(
                package="edgeai_ros",
                executable="image_publisher",
                name="edgeai_image_publisher",
                parameters=[{"image_dir": image_dir}],
                output="screen",
            ),
            Node(
                package="edgeai_ros",
                executable="adaptive_llm_node",
                name="adaptive_llm_node",
                parameters=[{"policy": policy, "deadline_ms": llm_deadline_ms}],
                output="screen",
                condition=IfCondition(enable_llm),
            ),
            Node(
                package="edgeai_ros",
                executable="edgeai_logger",
                name="edgeai_llm_logger",
                parameters=[{"latency_topic": "/llm_metrics", "csv_path": llm_csv_path}],
                output="screen",
                condition=IfCondition(enable_llm),
            ),
            Node(
                package="edgeai_ros",
                executable="prompt_publisher",
                name="edgeai_prompt_publisher",
                output="screen",
                condition=IfCondition(enable_llm),
            ),
        ]
    )

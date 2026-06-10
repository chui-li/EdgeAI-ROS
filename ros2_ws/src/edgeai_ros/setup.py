from glob import glob

from setuptools import find_packages, setup

package_name = "edgeai_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="EdgeAI-ROS Maintainers",
    maintainer_email="edgeai-ros@example.com",
    description="OS-aware adaptive ROS 2 runtime for RepViT inference under edge constraints.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "image_publisher = edgeai_ros.image_publisher:main",
            "prompt_publisher = edgeai_ros.prompt_publisher:main",
            "adaptive_repvit_node = edgeai_ros.adaptive_repvit_node:main",
            "adaptive_llm_node = edgeai_ros.adaptive_llm_node:main",
            "edgeai_logger = edgeai_ros.edgeai_logger:main",
            "stress_node = edgeai_ros.stress_node:main",
        ],
    },
)

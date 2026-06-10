FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-opencv \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    timm==0.5.4 \
    transformers==4.45.2 \
    psutil \
    pandas \
    matplotlib \
    opencv-python-headless \
    numpy

WORKDIR /workspace/EdgeAI-ROS

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV DEBIAN_FRONTEND=noninteractive \
    AI_RC_CAR_HOME=/opt/ai-rc-car

# Install system dependencies (including Python pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    liblapack-dev \
    libopenblas-base \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-pandas \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libopenmpi-dev \
    libomp-dev \
    zlib1g-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-opencv \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (now that pip3 is available)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install additional dependencies for PyTorch and TorchVision
RUN pip3 install --no-cache-dir future mock testresources Cython setuptools==58.3.0

# Install NumPy (required for building TorchVision)
RUN pip3 install --no-cache-dir numpy==1.19.4

# Install Python Dependencies
RUN pip3 install --no-cache-dir \
    Cython \
    pyyaml \
    cloudpickle==1.6.0 \
    protobuf==3.19.6 \
    posix_ipc~=1.0.4 \
    "Pillow==8.4.0" \
    "matplotlib==3.3.4" \
    "ipywidgets==7.6.5" \
    "traitlets>=4.3.3" \
    "Adafruit-MotorHAT>=1.4.0" \
    "Adafruit-SSD1306>=1.6.2" \
    "tensorboard>=2.8.0" \
    "gym==0.19.0" \
    "stable-baselines3==1.3.0" \
    git+https://github.com/tawnkramer/gym-donkeycar.git@v21.12.11 \
    git+https://github.com/NVIDIA-AI-IOT/jetbot.git@v0.4.3 \
    ipython \
    torchsummary \
    git+https://github.com/NVIDIA-AI-IOT/jetcam.git \
    git+https://github.com/NVIDIA-AI-IOT/jetracer.git

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR ${AI_RC_CAR_HOME}

# 4. Copy project files
COPY . ${AI_RC_CAR_HOME}

# Ensure Python training deps are present in the final image (usable inside training container).
# Note: torch here is a generic aarch64 wheel; if you want NVIDIA's optimized wheel, replace accordingly.
RUN pip3 install --no-cache-dir \
    gym==0.19.0 \
    stable-baselines3==1.3.0 \
    cloudpickle==1.6.0

# Install JetPack-specific CUDA-enabled torch wheel (do not use CPU-only PyPI build).
RUN pip3 install --no-cache-dir \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 \
    'torch==1.10.0+nv22.02'

RUN pip3 install --no-cache-dir --no-deps -e ${AI_RC_CAR_HOME}

EXPOSE 8888

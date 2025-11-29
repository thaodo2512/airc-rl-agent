# Base image can be swapped at build time (e.g., --build-arg BASE_IMAGE=python:3.8-slim for CPU-only).
ARG BASE_IMAGE=pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        wget \
        git \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU by default; for GPU base, skip this and use the image's pre-installed version)
RUN if [ "${BASE_IMAGE}" = "python:3.8-slim" ]; then \
        pip install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    fi

WORKDIR /workspace/airc-rl-agent
COPY . /workspace/airc-rl-agent

# Install dependencies manually to resolve conflicts, then install package without deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install Cython \
    && pip install gym==0.17.3 stable_baselines3==1.1.0 numpy scipy cloudpickle pyglet pillow pandas matplotlib h5py tensorboard \
    && pip install opencv-python \
    && pip install "gym-donkeycar @ git+https://github.com/tawnkramer/gym-donkeycar.git@v21.07.24#egg=gym-donkeycar" \
    && pip install --no-deps .

# Download pre-trained VAE model (adjust URL if needed; this is from repo references)
RUN wget "https://drive.google.com/uc?export=download&id=19r1yuwiRGGV-BjzjoCzwX8zmA8ZKFNcC" -O vae.torch

ENTRYPOINT ["racer"]
CMD ["--help"]

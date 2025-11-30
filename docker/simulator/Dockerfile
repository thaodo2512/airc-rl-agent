# Default to a CUDA 12.x PyTorch runtime that supports NVIDIA L4 GPUs (override BASE_IMAGE if needed).
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
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

# Pin NumPy to 1.26.x (avoids ABI breaks with PyTorch wheels) and install scientific deps via pip to sidestep slow conda solves
RUN pip install --no-cache-dir numpy==1.26.4 scipy h5py pandas matplotlib

# Install PyTorch (CPU by default; for GPU base, skip this and use the image's pre-installed version)
RUN if [ "${BASE_IMAGE}" = "python:3.8-slim" ]; then \
        pip install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    fi

WORKDIR /workspace/airc-rl-agent
COPY . /workspace/airc-rl-agent

# Install dependencies manually to resolve conflicts, then install package without deps
# No need for NumPy pip reinstall—handled by Conda above
RUN set -eux; \
    pip install --upgrade pip setuptools wheel; \
    pip install Cython; \
    pip install gym==0.17.3 stable_baselines3==1.1.0 cloudpickle pyglet pillow tensorboard; \
    pip install "opencv-python<4.12.0"; \
    pip install "gym-donkeycar @ git+https://github.com/tawnkramer/gym-donkeycar.git@v21.07.24#egg=gym-donkeycar"; \
    pip install --no-deps .; \
    pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Download pre-trained VAE model with confirmation handling (for Google Drive files)
RUN rm -rf vae.torch && \
    FILE_ID=19r1yuwiRGGV-BjzjoCzwX8zmA8ZKFNcC && \
    wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > /tmp/confirm && \
    CONFIRM=$(cat /tmp/confirm) && \
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID" -O vae.torch && \
    rm -rf /tmp/cookies.txt /tmp/confirm && \
    python - "$PWD/vae.torch" <<'PY'
import sys, torch
path = sys.argv[1]
sd = torch.load(path, map_location='cpu')
needs = any(k.startswith('decoder.7') for k in sd) and not all(k in sd for k in ('out1.0.weight','out1.0.bias','out2.0.weight','out2.0.bias'))
if needs:
    w = sd.get('decoder.7.weight')
    b = sd.get('decoder.7.bias')
    if w is not None and b is not None:
        state = sd
        state['out1.0.weight'] = state['decoder.7.weight']
        state['out1.0.bias'] = state['decoder.7.bias']
        state['out2.0.weight'] = state.pop('decoder.7.weight')
        state['out2.0.bias'] = state.pop('decoder.7.bias')
        torch.save(state, path)
    else:
        torch.save(sd, path)
PY

ENTRYPOINT ["racer"]
CMD ["--help"]

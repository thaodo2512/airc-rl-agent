#!/bin/bash
set -euo pipefail

echo "** Install requirement"
apt-get update
apt-get install -y --no-install-recommends liblapack-dev python3-scipy libfreetype6-dev python3-pandas
pip3 install --no-cache-dir --upgrade pip
pip3 install --no-cache-dir \
    stable-baselines3==1.3.0 \
    gym==0.19.0 \
    tensorboard>=2.8.0 \
    jetbot>=0.4.3 \
    Adafruit-MotorHAT>=1.4.0 \
    Adafruit-SSD1306>=1.6.2 \
    traitlets>=4.3.3 \
    opencv-python==4.5.5.64 \
    numpy==1.19.5 \
    ipywidgets==7.6.5 \
    Pillow==8.4.0 \
    matplotlib==3.3.4 \
    protobuf==3.19.6 \
    cloudpickle==1.6.0 \
    Cython \
    pyyaml \
    posix_ipc~=1.0.4 \
    git+https://github.com/tawnkramer/gym-donkeycar.git@v22.03.24#egg=gym-donkeycar

echo "** Building..."
pip3 install --no-cache-dir --no-deps '.[jetpack]'

echo "** Install learning_racer successfully"
echo "** Bye :)"

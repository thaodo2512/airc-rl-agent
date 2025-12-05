#!/bin/bash
set -e

# --- CONFIGURATION ---
# We pin these versions because official Jetbot images for 32.7.6 do not exist.
# We will use the 32.4.4 container on your 32.7.6 system.
export JETBOT_VERSION="0.4.2"
export BASE_L4T_VERSION="32.4.4"

# We can still label the output image with your actual system version if you want,
# or just keep it consistent with the base. Let's use the base to avoid confusion.
export OUTPUT_TAG="32.4.4"

export JETBOT_DOCKER_REMOTE="jetbot"

echo "---------------------------------------"
echo "Building learning_racer for Jetson Nano"
echo "Base Image: $JETBOT_DOCKER_REMOTE/jetbot:jupyter-$JETBOT_VERSION-$BASE_L4T_VERSION"
echo "Target Tag: learning_racer:$OUTPUT_TAG"
echo "---------------------------------------"

# --- BUILD ---
# We force the BASE_IMAGE argument to match the image we pulled in Step 1
sudo docker build \
    --build-arg BASE_IMAGE=$JETBOT_DOCKER_REMOTE/jetbot:jupyter-"$JETBOT_VERSION"-"$BASE_L4T_VERSION" \
    -t learning_racer:"$OUTPUT_TAG" \
    -f Dockerfile \
    ../..

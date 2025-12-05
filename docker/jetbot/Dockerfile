ARG BASE_IMAGE=jetbot/jetbot:jupyter-0.4.2-32.4.4
FROM ${BASE_IMAGE}

# --- ADD THIS SECTION ---
# Fix for "file exists" error: 
# Remove the old baked-in Tegra libraries so the Host (Jetpack 4.6) 
# can mount its newer drivers without conflict.
RUN rm -rf /usr/lib/aarch64-linux-gnu/tegra/* \
    && rm -rf /usr/lib/aarch64-linux-gnu/libnvidia*
# ------------------------

ENV AI_RC_CAR_HOME=/opt/ai-rc-car
COPY . ${AI_RC_CAR_HOME}
RUN cd ${AI_RC_CAR_HOME} && ls && sh install_jetpack_docker.sh

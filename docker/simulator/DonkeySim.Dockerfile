ARG UBUNTU_VERSION=20.04
FROM ubuntu:${UBUNTU_VERSION}

ARG DONKEYSIM_VERSION=v21.07.24
ARG DONKEYSIM_ZIP=DonkeySimLinux.zip

ENV DEBIAN_FRONTEND=noninteractive \
    DONKEYSIM_PORT=9091 \
    DONKEYSIM_HEADLESS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxi6 \
        libpulse0 \
        libunwind8 \
        libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/donkeysim
RUN wget -q "https://github.com/tawnkramer/gym-donkeycar/releases/download/${DONKEYSIM_VERSION}/${DONKEYSIM_ZIP}" -O /tmp/DonkeySim.zip \
    && unzip -q /tmp/DonkeySim.zip -d /opt/donkeysim \
    && rm /tmp/DonkeySim.zip \
    && chmod +x /opt/donkeysim/DonkeySimLinux/donkey_sim.x86_64 || true \
    && chmod +x /opt/donkeysim/DonkeySimLinux/DonkeySim.x86_64 || true

COPY run-donkey.sh /usr/local/bin/run-donkey.sh
RUN chmod +x /usr/local/bin/run-donkey.sh

EXPOSE 9091
ENTRYPOINT ["/usr/local/bin/run-donkey.sh"]

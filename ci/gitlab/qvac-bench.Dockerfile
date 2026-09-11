ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EXTRA_PACKAGES=""
ENV DEBIAN_FRONTEND=noninteractive

# Vulkan SDK from LunarG instead of Ubuntu's packages: the apt glslc is outdated
# and ggml-vulkan bakes its output into the binaries at build time
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg ca-certificates && \
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/lunarg.gpg && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git curl \
        python3 python3-venv \
        vulkan-sdk \
        ${EXTRA_PACKAGES} && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-qvac-bench.txt /tmp/requirements-qvac-bench.txt

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-qvac-bench.txt && \
    rm /tmp/requirements-qvac-bench.txt

ENV PATH=/opt/venv/bin:$PATH

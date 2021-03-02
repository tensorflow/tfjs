# Official emsdk docker: https://hub.docker.com/r/emscripten/emsdk
FROM emscripten/emsdk:2.0.14 AS emscripten_base

# Install cloud-sdk
ARG CLOUD_SDK_VERSION=305.0.0
ENV CLOUD_SDK_VERSION=$CLOUD_SDK_VERSION
ENV CLOUDSDK_PYTHON=python3
ENV PATH "$PATH:/opt/google-cloud-sdk/bin/"
RUN apt-get update -qqy && apt-get install -qqy \
        curl \
        gcc \
        python \
        python3 \
        python3-setuptools \
        python3-dev \
        python3-pip \
        apt-transport-https \
        lsb-release \
        openssh-client \
        git \
        gnupg \
        file && \
    pip3 install -U crcmod && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y google-cloud-sdk=${CLOUD_SDK_VERSION}-0 && \
    gcloud config set core/disable_usage_reporting true && \
    gcloud config set component_manager/disable_update_check true && \
    gcloud config set metrics/environment github_docker_image && \
    gcloud --version

RUN git config --system credential.'https://source.developers.google.com'.helper gcloud.sh

VOLUME ["/root/.config"]

# Install yarn
RUN npm install -g yarn

# Install virtualenv
RUN pip3 install virtualenv

# Install absl
RUN pip3 install absl-py

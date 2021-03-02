 # Official emsdk docker: https://hub.docker.com/r/emscripten/emsdk
 FROM emscripten/emsdk:2.0.14

 # Install yarn
 RUN npm install -g yarn

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
    pip3 install -U absl-py

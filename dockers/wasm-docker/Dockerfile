 # Official emsdk docker: https://hub.docker.com/r/emscripten/emsdk
FROM emscripten/emsdk:2.0.14

# Install yarn
RUN npm install -g yarn

RUN apt-get update -qqy && apt-get install -qqy \
        gcc \
        python3 \
        python3-pip \
        python \
        python-pip \
        file

# Install absl
RUN pip3 install -U absl-py

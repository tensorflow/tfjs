#!/bin/bash

source emscripten_toolchain/env.sh

exec python3 emscripten_toolchain/link_wrapper.py "$@"

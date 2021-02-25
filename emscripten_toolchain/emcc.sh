#!/bin/bash

source emscripten_toolchain/env.sh

exec python3 external/emscripten/emscripten/emcc.py "$@"

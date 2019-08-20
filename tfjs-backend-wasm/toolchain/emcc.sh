#!/bin/bash

set -euo pipefail

export LLVM_ROOT='external/emscripten_clang'
export EMSCRIPTEN_NATIVE_OPTIMIZER='external/emscripten_clang/optimizer'
export BINARYEN_ROOT='external/emscripten_clang/'
export NODE_JS=''
export EMSCRIPTEN_ROOT='external/emscripten_toolchain'
export SPIDERMONKEY_ENGINE=''
export EM_EXCLUSIVE_CACHE_ACCESS=1
export EMCC_SKIP_SANITY_CHECK=1
export EMCC_WASM_BACKEND=0

mkdir -p "tmp/emscripten_cache"

export EM_CACHE="tmp/emscripten_cache"
export TEMP_DIR="tmp"

# Prepare the cache content so emscripten doesn't keep rebuilding it
cp -r toolchain/emscripten_cache/* tmp/emscripten_cache

# Run emscripten to compile and link
python external/emscripten_toolchain/emcc.py "$@"

# Remove the first line of .d file
find . -name "*.d" -exec sed -i '2d' {} \;

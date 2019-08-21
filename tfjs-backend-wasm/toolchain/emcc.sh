#!/bin/bash

set -euo pipefail

export LLVM_ROOT='external/emsdk/emsdk/fastcomp/fastcomp/bin'
export BINARYEN_ROOT='external/emsdk/fastcomp'
export EMSCRIPTEN_NATIVE_OPTIMIZER='external/emsdk/emsdk/fastcomp/bin/optimizer'
export NODE_JS='external/emsdk/node/8.9.1_64bit/bin/node'
export SPIDERMONKEY_ENGINE=''
export V8_ENGINE=''
export TEMP_DIR="tmp"
export COMPILER_ENGINE=$NODE_JS
export JS_ENGINES=$NODE_JS
export EMSCRIPTEN_ROOT='external/emsdk/fastcomp'

export EM_EXCLUSIVE_CACHE_ACCESS=1
export EMCC_SKIP_SANITY_CHECK=1
export EMCC_WASM_BACKEND=0

# TODO(smilkov): Figure out the gnu sed vs mac os sed
export PATH="/usr/local/opt/gnu-sed/libexec/gnubin:$PATH"

# Run emscripten to compile and link
python2 external/emsdk/emsdk/fastcomp/emscripten/emcc.py "$@"

# Remove the first line of .d file
find . -name "*.d" -exec gsed -i '2d' {} \;

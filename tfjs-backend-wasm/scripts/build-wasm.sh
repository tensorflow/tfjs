#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Halt if a single command errors
set -e

# Echo every command being executed
set -x

# Default build.
yarn bazel build $BAZEL_REMOTE -c opt //tfjs-backend-wasm/src/cc:tfjs-backend-wasm
# The typescript code and karma config expect the output of emscripten to be in
# wasm-out/ so we copy the bazel output there.
cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm/tfjs-backend-wasm.js \
      ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm/tfjs-backend-wasm.wasm \
      ../wasm-out/

if [[ "$1" != "--dev" ]]; then
  # SIMD and threaded + SIMD builds.
  yarn bazel build $BAZEL_REMOTE -c opt --copt="-msimd128" //tfjs-backend-wasm/src/cc:tfjs-backend-wasm-simd \
    //tfjs-backend-wasm/src/cc:tfjs-backend-wasm-threaded-simd

  # Copy SIMD
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-simd/tfjs-backend-wasm.wasm \
    ../wasm-out/tfjs-backend-wasm-simd.wasm

  # Copy threaded
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd/tfjs-backend-wasm-threaded-simd.js \
    ../wasm-out/tfjs-backend-wasm-threaded-simd.js
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd/tfjs-backend-wasm-threaded-simd.worker.js \
    ../wasm-out/tfjs-backend-wasm-threaded-simd.worker.js
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd/tfjs-backend-wasm-threaded-simd.wasm \
    ../wasm-out/tfjs-backend-wasm-threaded-simd.wasm

  node ./create-worker-module.js
  node ./patch-threaded-simd-module.js
fi

mkdir -p ../dist
# Only copying binaries into dist because the js modules get bundled.
cp ../wasm-out/*.wasm ../dist/

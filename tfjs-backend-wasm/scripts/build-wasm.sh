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
yarn bazel build -c opt //tfjs-backend-wasm/src/cc:tfjs-backend-wasm.js --config=wasm
# The typescript code and karma config expect the output of emscripten to be in
# wasm-out/ so we copy the bazel output there.
cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm.js \
      ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm.wasm \
      ../wasm-out/

if [[ "$1" != "--dev" ]]; then
  # SIMD build.
  yarn bazel build -c opt //tfjs-backend-wasm/src/cc:tfjs-backend-wasm-simd.js --config=wasm --copt="-msimd128"
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-simd.wasm \
        ../wasm-out/

  # Threaded + SIMD build.
  yarn bazel build -c opt //tfjs-backend-wasm/src/cc:tfjs-backend-wasm-threaded-simd.js --config=wasm --copt="-pthread" --copt="-msimd128"
  cp -f ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd.js \
        ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd.worker.js \
        ../../dist/bin/tfjs-backend-wasm/src/cc/tfjs-backend-wasm-threaded-simd.wasm \
        ../wasm-out/

  node ./create-worker-module.js
  node ./patch-threaded-simd-module.js
fi

mkdir -p ../dist
# Only copying binaries into dist because the js modules get bundled.
cp ../wasm-out/*.wasm ../dist/

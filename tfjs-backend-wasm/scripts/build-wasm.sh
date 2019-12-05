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

set -e

yarn bazel build -c opt //src/cc:tfjs-backend-wasm.js --config=wasm
# The typescript code and karma config expect the output of emscripten to be in
# wasm-out/ so we copy the bazel output there.
cp -f bazel-bin/src/cc/tfjs-backend-wasm.js \
      bazel-bin/src/cc/tfjs-backend-wasm.wasm \
      wasm-out/

cp wasm-out/*.wasm dist/

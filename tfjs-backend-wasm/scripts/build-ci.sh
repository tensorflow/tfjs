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

# Install emsdk
git clone --depth=1 --single-branch https://github.com/emscripten-core/emsdk.git
cd emsdk
# Need to tell emsdk where to write the .emscripten file.
export HOME='/root'
./emsdk install 1.39.1
./emsdk activate 1.39.1
source ./emsdk_env.sh
cd ..

yarn tsc

./scripts/build-wasm.sh

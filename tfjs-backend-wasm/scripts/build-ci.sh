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

git clone --depth=1 --single-branch https://github.com/emscripten-core/emsdk.git

cd emsdk
# Need to tell emsdk where to write the .emscripten file.
export HOME='/root'

# Install emsdk with up to 1 retry.
for i in $(seq 0 1)
do
  # Wait for 15 seconds then retry.
  [ $i -gt 0 ] && echo "Retry in 15 seconds, count: $i" && sleep 15
  # If install is successful, $? will hold 0 and execution will break from the
  # loop.
  ./emsdk install 1.39.1 && break
done

./emsdk activate 1.39.1
source ./emsdk_env.sh
cd ..

yarn tsc

./scripts/build-wasm.sh

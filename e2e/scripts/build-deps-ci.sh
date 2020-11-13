#!/usr/bin/env bash
# Copyright 2020 Google LLC
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
# ==============================================================================

set -e

yarn build-core-ci
yarn build-backend-cpu-ci
yarn build-backend-webgl-ci
yarn build-layers-ci
yarn build-converter-ci
yarn build-data-ci
yarn build-union-ci
yarn build-node-ci

if [[ "$NIGHTLY" = true || "$RELEASE" = true ]]; then
  # Build the wasm backend
  yarn build-backend-wasm-ci
fi

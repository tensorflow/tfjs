#!/usr/bin/env bash
# Copyright 2021 Google LLC. All Rights Reserved.
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

yarn rimraf dist/
yarn

yarn build
yarn rollup -c --visualize --npm

# Use minified files for miniprogram
mkdir dist/miniprogram
cp dist/tf-backend-webgpu.min.js dist/miniprogram/index.js
cp dist/tf-backend-webgpu.min.js.map dist/miniprogram/index.js.map

echo "Stored standalone library at dist/tf-backend-webgpu(.min).js"

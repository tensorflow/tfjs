#!/usr/bin/env bash
# Copyright 2018 Google LLC. All Rights Reserved.
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

# Clean the build.
yarn rimraf dist/
yarn rimraf deps/
yarn rimraf lib/

# Download the tensorflow headers and lib.
yarn install
# Build the pre-built addon. Do not publish it yet.
yarn build-addon-from-source

tsc --sourceMap false
# Manual copy src/proto/api_pb.js until both allowJs and declaration are
# supported in tsconfig: https://github.com/microsoft/TypeScript/pull/32372
mkdir -p dist/proto && cp src/proto/api_pb.js dist/proto/api_pb.js

npm pack

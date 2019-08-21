#!/usr/bin/env bash
# Copyright 2018 Google Inc. All Rights Reserved.
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

# The binding builds with a symlink by default, for NPM packages change the
# download option to move libtensorflow next to the prebuilt binary.
sed -i -e 's/symlink/move/' binding.gyp

# Build GPU:
sed -i -e 's/tfjs-node"/tfjs-node-gpu"/' package.json
sed -i -e 's/install.js"/install.js gpu download"/' package.json
rimraf deps/
rimraf dist/
rimraf lib/
yarn build-addon "$1"
yarn prep
tsc --sourceMap false
# This produces a tarball that will later be used by `npm publish`.
npm pack

# Revert GPU changes:
sed -i -e 's/move/symlink/' binding.gyp
sed -i -e 's/tfjs-node-gpu"/tfjs-node"/' package.json
sed -i -e 's/install.js gpu download"/install.js"/' package.json

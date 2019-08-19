#!/usr/bin/env bash
# Copyright 2017 Google Inc. All Rights Reserved.
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

PACKAGE_JSON_FILE="$1/package.json"
if ! test -f "$PACKAGE_JSON_FILE"; then
  echo "$PACKAGE_JSON_FILE does not exist."
  echo "Please pass the package name as the first argument."
  exit 1
fi

yarn rimraf $1/dist/

cd $1
yarn

yarn build-npm
yarn rollup -c --visualize

# Use minified files for miniprogram
mkdir dist/miniprogram
MIN_NAME=${1/tfjs/tf}
cp dist/$MIN_NAME.min.js dist/miniprogram/index.js
cp dist/$MIN_NAME.min.js.map dist/miniprogram/index.js.map

echo "Stored standalone library at dist/$MIN_NAME(.min).js"

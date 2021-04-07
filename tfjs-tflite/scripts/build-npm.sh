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

DEPS_DIR="./deps"
if [ ! -d ${DEPS_DIR} ] || [ ! "$(ls -A ${DEPS_DIR})" ]; then
    echo "ERROR: tfweb not downloaded yet. Run './scripts/download-tfweb.sh {version}' first."
    exit 1
fi

yarn rimraf dist/
mkdir -p dist
yarn

cp ${DEPS_DIR}/* dist/
yarn build
yarn rollup -c --visualize --npm

echo "Stored standalone library at dist/tf-tflite(.min).js"

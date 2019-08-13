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

function print_status() {
  name=$1
  code=$2
  echo -e "$name:" $([[ $code -eq 0 ]] && echo 'PASS' || echo 'FAIL') $3
}

echo '######################'
echo 'Nightly build or version.ts was modified.'
echo 'Testing layers/converter/node/data against tfjs-core@master.'
echo '######################'
yarn build && yarn build-test-snippets && yarn yalc publish

echo 'Cloning layers'
git clone https://github.com/tensorflow/tfjs-layers.git --depth 1
cd tfjs-layers
yarn && yarn link-local '@tensorflow/tfjs-core'
yarn test-ci
LAYERS_EXIT_CODE=$?

cd ..
echo 'Cloning node'
git clone https://github.com/tensorflow/tfjs-node.git --depth 1
cd tfjs-node
yarn && yarn link-local '@tensorflow/tfjs-core'
yarn test-ci
NODE_EXIT_CODE=$?

cd ..
echo 'Cloning converter'
git clone https://github.com/tensorflow/tfjs-converter.git --depth 1
cd tfjs-converter
yarn && yarn link-local '@tensorflow/tfjs-core'
yarn test-ci
CONVERTER_EXIT_CODE=$?

cd ..
echo 'Cloning data'
git clone https://github.com/tensorflow/tfjs-data.git --depth 1
cd tfjs-data
yarn && yarn link-local '@tensorflow/tfjs-core'
yarn test-ci
DATA_EXIT_CODE=$?

echo '==== INTEGRATION TEST RESULTS ===='
print_status "tfjs-layers" "$LAYERS_EXIT_CODE"
print_status "tfjs-node" "$NODE_EXIT_CODE"
print_status "tfjs-converter" "$CONVERTER_EXIT_CODE"
print_status "tfjs-data" "$DATA_EXIT_CODE"
echo '=================================='

RED='\033[0;31m'
BLUE='\e[34m'
NC='\033[0m' # No Color
FINAL_EXIT_CODE=$(($LAYERS_EXIT_CODE+$NODE_EXIT_CODE+$CONVERTER_EXIT_CODE+$DATA_EXIT_CODE))
[[ $FINAL_EXIT_CODE -eq 0 ]] && COLOR=$BLUE || COLOR=$RED
print_status "${COLOR}Final result" "$FINAL_EXIT_CODE" "$NC"
exit $FINAL_EXIT_CODE

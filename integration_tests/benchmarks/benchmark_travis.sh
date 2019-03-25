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

yarn
yarn lint

if [ "$TRAVIS_EVENT_TYPE" = cron ] && [[ $(node -v) = *v10* ]]
then
  # Run the first karma separately so it can download the BrowserStack binary
  # without conflicting with others.
  echo 'Run first karma.'
  yarn run-browserstack --browsers=bs_safari_mac --grep=matmul

  echo 'Use latest version of tfjs-core'
  git clone https://github.com/tensorflow/tfjs-core.git --depth 5
  cd tfjs-core
  HASH_CORE=`git rev-parse HEAD`
  rm -rf dist/ && yarn && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-core'

  echo 'Use latest version of tfjs-layers'
  git clone https://github.com/tensorflow/tfjs-layers.git --depth 5
  cd tfjs-layers
  HASH_LAYERS=`git rev-parse HEAD`
  rm -rf dist/ && yarn && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-layers'

  echo 'Use latest version of tfjs-converter'
  git clone https://github.com/tensorflow/tfjs-converter.git --depth 5
  cd tfjs-converter
  HASH_CONVERTER=`git rev-parse HEAD`
  rm -rf dist/ && yarn && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-converter'

  echo 'Use latest version of tfjs-data'
  git clone https://github.com/tensorflow/tfjs-data.git --depth 5
  cd tfjs-data
  HASH_DATA=`git rev-parse HEAD`
  rm -rf dist/ && yarn && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-data'

  npm-run-all -p -c --aggregate-output \
    "run-browserstack --travis --browsers=bs_ios_11 --grep=mobilenet --hashes='{\"CORE\":\"$HASH_CORE\",\"LAYERS\":\"$HASH_LAYERS\",\"CONVERTER\":\"$HASH_CONVERTER\"}'" \
    "run-browserstack --travis --browsers=bs_safari_mac --grep=models --hashes='{\"CORE\":\"$HASH_CORE\",\"LAYERS\":\"$HASH_LAYERS\",\"CONVERTER\":\"$HASH_CONVERTER\"}'" \
    "run-browserstack --travis --browsers=bs_chrome_mac --grep=models --hashes='{\"CORE\":\"$HASH_CORE\",\"LAYERS\":\"$HASH_LAYERS\",\"CONVERTER\":\"$HASH_CONVERTER\"}'" \
    "run-browserstack --travis --browsers=bs_ios_11 --grep=ops --hashes='{\"CORE\":\"$HASH_CORE\"}'" \
    "run-browserstack --travis --browsers=bs_safari_mac --grep=ops --hashes='{\"CORE\":\"$HASH_CORE\"}'" \
    "run-browserstack --travis --browsers=bs_chrome_mac --grep=ops --hashes='{\"CORE\":\"$HASH_CORE\"}'"
fi

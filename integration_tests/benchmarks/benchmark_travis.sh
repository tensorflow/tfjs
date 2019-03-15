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

if [ "$TRAVIS_EVENT_TYPE" = cron ] && [[ $(node -v) = *v10* ]]
then
  yarn
  yarn lint

  echo 'Use latest version of tfjs-core'
  git clone https://github.com/tensorflow/tfjs-core.git --depth 5
  cd tfjs-core
  rm -rf dist/ && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-core'

  echo 'Use latest version of tfjs-layers'
  git clone https://github.com/tensorflow/tfjs-layers.git --depth 5
  cd tfjs-layers
  rm -rf dist/ && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-layers'

  echo 'Use latest version of tfjs-node'
  git clone https://github.com/tensorflow/tfjs-node.git --depth 5
  cd tfjs-node
  rm -rf dist/ && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-node'

  echo 'Use latest version of tfjs-converter'
  git clone https://github.com/tensorflow/tfjs-converter.git --depth 5
  cd tfjs-converter
  rm -rf dist/ && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-converter'

  echo 'Use latest version of tfjs-data'
  git clone https://github.com/tensorflow/tfjs-data.git --depth 5
  cd tfjs-data
  rm -rf dist/ && yarn build && rollup -c && yalc push

  cd ..
  yarn link-local '@tensorflow/tfjs-data'

  karma start --firebaseKey $FIREBASE_KEY --travis \
    --singleRun --reporters='dots,karma-typescript,BrowserStack' \
    --hostname='bs-local.com' --browsers=bs_chrome_mac
fi

#!/usr/bin/env bash
# Copyright 2020 Google LLC. All Rights Reserved.
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

################################################################################
# NOTE: You must run this script from tfjs-node/
################################################################################

echo "=====Publish tfjs-core locally.====="
cd ../tfjs-core
yarn build & yarn yalc publish

echo "=====Publish tfjs-converter locally.====="
cd ../tfjs-converter
yarn build & yarn yalc publish

echo "=====Publish tfjs-data locally.====="
cd ../tfjs-data
yarn build & yarn yalc publish

echo "=====Publish tfjs-layers locally.====="
cd ../tfjs-layers
yarn build & yarn yalc publish

echo "=====Link package.====="
cd ../tfjs-node
yarn yalc link @tensorflow/tfjs-core & yarn yalc link @tensorflow/tfjs-converter & yarn yalc link @tensorflow/tfjs-data & yarn yalc link @tensorflow/tfjs-layers & yarn

echo "=====Start testing.====="
ts-node src/run_tests.ts

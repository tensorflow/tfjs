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

# Start in scripts/ even if run from root directory
cd "$(dirname "$0")"

# Go to e2e root
echo $PWD
cd ..
e2e_root_path=$PWD

cd custom_module/blazeface
yarn
# Ensure that we test against freshly generated custom modules.
rm -f ./custom_tfjs_blazeface/*.js
echo "npm version $(npm --version)"
yarn make-custom-tfjs-modules
# TODO(yassogba) once blazeface kernels are modularized in cpu
# switch the config to cpu and also run and test rollup bundle.
yarn webpack:full
yarn webpack:custom

echo $PWD
cd $e2e_root_path
echo $PWD
cd custom_module/dense_model
yarn
# Ensure that we test against freshly generated custom modules.
rm -f ./custom_tfjs/*.js
yarn make-custom-tfjs-modules

yarn webpack:full
yarn webpack:custom
yarn rollup:full
yarn rollup:custom

cd $e2e_root_path
echo $PWD
cd custom_module/universal_sentence_encoder
yarn
# Ensure that we test against freshly generated custom modules.
rm -f ./custom_tfjs/*.js
yarn make-custom-tfjs-modules

yarn webpack:full
yarn webpack:custom
yarn rollup:full
yarn rollup:custom
